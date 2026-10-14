// Multi-input BSGS baby-step MAC for Holmes's LinearOperator (uint64 RNS, N = 2^16).
//
// Holmes runs hoisted_mac_sum once per input ciphertext: every plaintext chunk (G giants x B babies rows,
// rows) is streamed from HBM once per input, repeating the plaintext reads.
// This kernel computes the partial sums of TI inputs at once from their (pre-rotated, non-ext) baby
// bases, so each plaintext element is read once per TI inputs and each baby element once per chunk:
//
//   res[c][g][i][limb][j] = sum_b pt[g * B + b][limb][j] * baby_i[b][c][limb][j]   (mod q_limb)
//
// Every product is reduced with the plaintext's Shoup constant (computed on the fly, exact), the
// reduced products are summed in 64 bits and folded back below p every 8 terms (9 p < 2^64 for the
// <= 60-bit primes), so the outputs are the exact residues of EasyFHE's batched_pairwise_mac. Modular helpers follow Sylph's
// easyfhe_ext/kernels/keyswitch.cu.
#include <cstdint>
#include <cuda_runtime.h>

namespace holmes_mac {

constexpr int kLogN = 16;
constexpr int kN = 1 << kLogN;
constexpr int kMaxInputs = 32;

struct u128 { uint64_t hi, lo; };

__device__ __forceinline__ u128 mul128(uint64_t a, uint64_t b) {
  u128 r; r.lo = a * b; r.hi = __umul64hi(a, b); return r;
}
__device__ __forceinline__ uint64_t barrett128(const u128 in, uint64_t prime, uint64_t ratio, uint64_t k) {
  u128 t1 = mul128(in.lo, ratio);
  u128 t2 = mul128(in.hi, ratio);
  asm("add.cc.u64 %0, %0, %1;" : "+l"(t1.hi) : "l"(t2.lo));
  asm("{addc.cc.u64 %0, %0, %1;}" : "+l"(t2.hi) : "l"((unsigned long)0));
  const unsigned shift = static_cast<unsigned>(k - 64);
  t1.hi >>= shift;
  t2.hi <<= 64 - shift;
  t1.hi += t2.hi;
  t1.hi *= prime;
  uint64_t res = in.lo - t1.hi;
  if (res >= prime) res -= prime;
  return res;
}
// a * b mod p with b_shoup = floor(b * 2^64 / p); result in [0, 2p)
__device__ __forceinline__ uint64_t shoup(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t prime) {
  const uint64_t hi = __umul64hi(b_shoup, a);
  return a * b - hi * prime;
}
__device__ __forceinline__ uint64_t shoup_full(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t prime) {
  uint64_t r = shoup(a, b, b_shoup, prime);
  return r >= prime ? r - prime : r;
}

// floor(w * 2^64 / p) for w < p, integer only: Barrett estimate (w * ratio) >> (k - 64) with
// ratio = floor(2^k / p), k = floor(log2 p) + 63, which is below the exact value by less than 4
// (w r / (p 2^(k-64)) < p / 2^(floor(log2 p) - 1) < 4), then the 128-bit remainder
// w 2^64 - est p in [0, 5p) is brought below p by three conditional steps (4p, 2p, p).
// (The double-precision version of Sylph's pts_shoup is 30+ FP64 ops: 1/64 rate on this card.)
__device__ __forceinline__ bool ge128(uint64_t hi, uint64_t lo, uint64_t bhi, uint64_t blo) {
  return hi > bhi || (hi == bhi && lo >= blo);
}
__device__ __forceinline__ uint64_t shoup_const(uint64_t w, uint64_t p, uint64_t ratio, unsigned shift) {
  const u128 wr = mul128(w, ratio);
  uint64_t est = (wr.lo >> shift) | (wr.hi << (64 - shift));   // (wr >> shift), shift in [1, 63]
  u128 ep = mul128(est, p);
  // rem = w * 2^64 - est * p  (>= 0, < 5p < 2^63)
  uint64_t lo = 0 - ep.lo;
  uint64_t hi = w - ep.hi - (ep.lo != 0 ? 1 : 0);
  const u128 p4 = mul128(p, 4), p2 = mul128(p, 2);
  if (ge128(hi, lo, p4.hi, p4.lo)) { est += 4; hi -= p4.hi + (lo < p4.lo ? 1 : 0); lo -= p4.lo; }
  if (ge128(hi, lo, p2.hi, p2.lo)) { est += 2; hi -= p2.hi + (lo < p2.lo ? 1 : 0); lo -= p2.lo; }
  if (ge128(hi, lo, 0, p)) { est += 1; }
  return est;
}

__device__ __forceinline__ uint64_t fold9(uint64_t v, uint64_t p, uint64_t p8) {
  if (v >= p8) v -= p8;
  if (v >= (p << 2)) v -= (p << 2);
  if (v >= (p << 1)) v -= (p << 1);
  if (v >= p) v -= p;
  return v;
}

// x mod p for a 128-bit x < 32 p^2 (sum of <= 16 products of residues below 2 p, p < 2^60): Barrett estimate
// q = floor(x * ratio / 2^k) with ratio = floor(2^k / p), k = floor(log2 p) + 63, which undershoots the true
// quotient by at most x / 2^k + 2 (59-bit primes: x < 2^123, k = 121, < 6; 60-bit primes: x < 2^125, k = 122,
// < 10), so x - q p < 10 p < 2^64 fits the low word and is brought below p by conditional subtractions of
// 8 p, 4 p, 2 p, p (16 p still fits in 64 bits for p < 2^60).
__device__ __forceinline__ uint64_t barrett_lazy(uint64_t hi, uint64_t lo, uint64_t p, uint64_t ratio, unsigned shift) {
  u128 t1 = mul128(lo, ratio);
  u128 t2 = mul128(hi, ratio);
  const uint64_t mid = t1.hi + t2.lo;
  const uint64_t top = t2.hi + (mid < t2.lo ? 1 : 0);
  // q = (top : mid) >> shift, shift = k - 64 in [1, 63]; q < 2^64 since x / p < 16 p < 2^64
  const uint64_t q = (mid >> shift) | (top << (64 - shift));
  uint64_t r = lo - q * p;   // exact low word of x - q p (< 16 p < 2^64 by the bound above; 32 p would overflow)
  const uint64_t p8 = p << 3, p4 = p << 2, p2 = p << 1;
  if (r >= p8) r -= p8;
  if (r >= p4) r -= p4;
  if (r >= p2) r -= p2;
  if (r >= p) r -= p;
  return r;
}

struct Inputs {
  const uint64_t* c0[kMaxInputs];
  const uint64_t* c1[kMaxInputs];
};

// grid: (kN / 256 * ceil(inputs / TI), limbs, ceil(G / TG)). Thread (limb, j) holds a TI x TG tile of
// (input, giant group) accumulators for both ciphertext components.  Per baby step b it reads TI baby residues
// and TG plaintext residues for TI x TG x 2 MACs, so the whole chunk streams baby_i once per G / TG group tiles
// and every plaintext row once per inputs / TI input tiles: traffic ~ inputs * G * (1 / TG + 1 / TI) elements
// per (b, limb, j) at 2 * TI * TG accumulators (square tiles minimise it for a register budget).  pts rows
// [G * nb][>= limbs][N] with row stride pstride; baby bases [nb][>= limbs][N], baby stride bstride (shared by
// c0 / c1 of every input); res [2][G][inputs][limbs][N] contiguous (so res[c][g] is a [inputs][limbs][N] batch
// for the giant step).
template <int TI, int TG>
__global__ void __launch_bounds__(256, (TI * TG <= 16 && TG <= 4) ? 2 : 1) mac_multi(uint64_t* __restrict__ res, const Inputs in,
                                                                        long long bstride, int nb, int inputs, int G,
                                                                        const uint64_t* __restrict__ pts, long long pstride,
                                                                        int limbs, const uint64_t* __restrict__ primes,
                                                                        const uint64_t* __restrict__ ratios,
                                                                        const uint64_t* __restrict__ ks) {
  const int ntile = (inputs + TI - 1) / TI;
  const int j = (blockIdx.x / ntile) * 256 + threadIdx.x, limb = blockIdx.y, i0 = (blockIdx.x % ntile) * TI;
  const int g0 = blockIdx.z * TG;
  const uint64_t p = primes[limb], ratio = ratios[limb], p8 = p << 3;
  const unsigned shift = static_cast<unsigned>(ks[limb] - 64);   // 58 or 59 for 59 / 60-bit primes
  const long long lj = (long long)limb * kN + j;
  uint64_t acc[TI][2][TG];
#pragma unroll
  for (int t = 0; t < TI; ++t)
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int g = 0; g < TG; ++g) acc[t][c][g] = 0;
  const uint64_t* cp[TI][2];
#pragma unroll
  for (int t = 0; t < TI; ++t) {
    const int i = i0 + t < inputs ? i0 + t : i0;   // clamp: duplicates are harmless, never written
    cp[t][0] = in.c0[i] + lj;
    cp[t][1] = in.c1[i] + lj;
  }
  const uint64_t* pp[TG];
#pragma unroll
  for (int g = 0; g < TG; ++g) {
    const int gg = g0 + g < G ? g0 + g : g0;   // clamp likewise
    pp[g] = pts + (long long)gg * nb * pstride + lj;
  }
  for (int b = 0; b < nb; ++b) {
    uint64_t w[TG], wsh[TG];
    const long long po = (long long)b * pstride;
#pragma unroll
    for (int g = 0; g < TG; ++g) {
      w[g] = pp[g][po];
      wsh[g] = shoup_const(w[g], p, ratio, shift);
    }
    const long long bo = (long long)b * bstride;
#pragma unroll
    for (int t = 0; t < TI; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint64_t v = cp[t][c][bo];
#pragma unroll
        for (int g = 0; g < TG; ++g) acc[t][c][g] += shoup_full(v, w[g], wsh[g], p);
      }
    }
    if ((b & 7) == 7) {
      // acc < 9 p < 2^64 (p < 2^60): back to [0, p) with four conditional subtractions
#pragma unroll
      for (int t = 0; t < TI; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
          for (int g = 0; g < TG; ++g) acc[t][c][g] = fold9(acc[t][c][g], p, p8);
    }
  }
  const long long is = (long long)limbs * kN, gs = (long long)inputs * is;   // res [2][G][inputs][limbs][N]
#pragma unroll
  for (int t = 0; t < TI; ++t) {
    const int i = i0 + t;
    if (i >= inputs) break;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint64_t* r = res + (long long)c * G * gs + (long long)i * is + lj;
#pragma unroll
      for (int g = 0; g < TG; ++g)
        if (g0 + g < G) r[(long long)(g0 + g) * gs] = fold9(acc[t][c][g], p, p8);
    }
  }
}

// Same tile as mac_multi, but every accumulator is a 128-bit lazy sum: a product costs one 64 x 64 -> 128
// multiply and a 128-bit add (no Shoup constant, no per-product reduction); the sums are reduced with one
// Barrett per 16 baby steps (barrett_lazy) and once at the end.  Residues are identical to mac_multi (both
// are the exact sum mod p); the accumulators take twice the registers, so the tiles are half as large.
template <int TI, int TG>
__global__ void __launch_bounds__(256, (TI * TG <= 8) ? 2 : 1) mac_multi_lazy(uint64_t* __restrict__ res, const Inputs in,
                                                                           long long bstride, int nb, int inputs, int G,
                                                                           const uint64_t* __restrict__ pts, long long pstride,
                                                                           int limbs, const uint64_t* __restrict__ primes,
                                                                           const uint64_t* __restrict__ ratios,
                                                                           const uint64_t* __restrict__ ks) {
  const int ntile = (inputs + TI - 1) / TI;
  const int j = (blockIdx.x / ntile) * 256 + threadIdx.x, limb = blockIdx.y, i0 = (blockIdx.x % ntile) * TI;
  const int g0 = blockIdx.z * TG;
  const uint64_t p = primes[limb], ratio = ratios[limb];
  const unsigned shift = static_cast<unsigned>(ks[limb] - 64);
  const long long lj = (long long)limb * kN + j;
  uint64_t lo[TI][2][TG], hi[TI][2][TG];
#pragma unroll
  for (int t = 0; t < TI; ++t)
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int g = 0; g < TG; ++g) { lo[t][c][g] = 0; hi[t][c][g] = 0; }
  const uint64_t* cp[TI][2];
#pragma unroll
  for (int t = 0; t < TI; ++t) {
    const int i = i0 + t < inputs ? i0 + t : i0;
    cp[t][0] = in.c0[i] + lj;
    cp[t][1] = in.c1[i] + lj;
  }
  const uint64_t* pp[TG];
#pragma unroll
  for (int g = 0; g < TG; ++g) {
    const int gg = g0 + g < G ? g0 + g : g0;
    pp[g] = pts + (long long)gg * nb * pstride + lj;
  }
  for (int b = 0; b < nb; ++b) {
    uint64_t w[TG];
    const long long po = (long long)b * pstride;
#pragma unroll
    for (int g = 0; g < TG; ++g) w[g] = pp[g][po];
    const long long bo = (long long)b * bstride;
#pragma unroll
    for (int t = 0; t < TI; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint64_t v = cp[t][c][bo];
#pragma unroll
        for (int g = 0; g < TG; ++g) {
          const uint64_t plo = v * w[g], phi = __umul64hi(v, w[g]);
          const uint64_t s = lo[t][c][g] + plo;   // carry flags do not survive between asm statements: portable add
          hi[t][c][g] += phi + (s < plo ? 1 : 0);
          lo[t][c][g] = s;
        }
      }
    }
    if ((b & 15) == 15 && b + 1 < nb) {
#pragma unroll
      for (int t = 0; t < TI; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
          for (int g = 0; g < TG; ++g) {
            lo[t][c][g] = barrett_lazy(hi[t][c][g], lo[t][c][g], p, ratio, shift);
            hi[t][c][g] = 0;
          }
    }
  }
  const long long is = (long long)limbs * kN, gs = (long long)inputs * is;
#pragma unroll
  for (int t = 0; t < TI; ++t) {
    const int i = i0 + t;
    if (i >= inputs) break;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint64_t* r = res + (long long)c * G * gs + (long long)i * is + lj;
#pragma unroll
      for (int g = 0; g < TG; ++g)
        if (g0 + g < G) r[(long long)(g0 + g) * gs] = barrett_lazy(hi[t][c][g], lo[t][c][g], p, ratio, shift);
    }
  }
}

// Gathered MAC for the attention sparse BSGS scheduler (HOLMES_SPARSE_FUSED): group k sums the rows
// seg[k] <= r < seg[k + 1], row r multiplying plaintext row prow[r] by baby idx[r] of every input:
//   res[c][k][i][limb][j] = sum_{seg[k] <= r < seg[k+1]} pts[prow[r]][limb][j] * baby_i[idx[r]][c][limb][j]   (mod q_limb)
// res [2][K][inputs][limbs][N] (res[c][k] is a contiguous [inputs][limbs][N] batch: the giant step's batched
// key switch reads res[c][:nk] as one [nk * inputs][limbs][N] batch).  Replaces the Python loop's per-group baby
// selection copy (torch.cat of the group's babies) + grouped_pairwise_mac (K * inputs launches per encode batch)
// with one launch; lazy 128-bit accumulation as mac_multi_lazy (exact residues).  Baby bases [nb][>= limbs][N]
// (item stride bstride) per input; rows of one group are <= 2^31 and every idx[r] < nb.
template <int TI>
__global__ void __launch_bounds__(256, 2) mac_gather(uint64_t* __restrict__ res, const Inputs in, long long bstride,
                                                     const int* __restrict__ idx, const int* __restrict__ prow,
                                                     const int* __restrict__ seg,
                                                     int inputs, int K, const uint64_t* __restrict__ pts,
                                                     long long pstride, int limbs,
                                                     const uint64_t* __restrict__ primes,
                                                     const uint64_t* __restrict__ ratios,
                                                     const uint64_t* __restrict__ ks) {
  const int ntile = (inputs + TI - 1) / TI;
  const int j = (blockIdx.x / ntile) * 256 + threadIdx.x, limb = blockIdx.y, i0 = (blockIdx.x % ntile) * TI;
  const int k = blockIdx.z;
  const int r0 = seg[k], r1 = seg[k + 1];
  const uint64_t p = primes[limb], ratio = ratios[limb];
  const unsigned shift = static_cast<unsigned>(ks[limb] - 64);
  const long long lj = (long long)limb * kN + j;
  uint64_t lo[TI][2], hi[TI][2];
#pragma unroll
  for (int t = 0; t < TI; ++t)
#pragma unroll
    for (int c = 0; c < 2; ++c) { lo[t][c] = 0; hi[t][c] = 0; }
  const uint64_t* cp[TI][2];
#pragma unroll
  for (int t = 0; t < TI; ++t) {
    const int i = i0 + t < inputs ? i0 + t : i0;
    cp[t][0] = in.c0[i] + lj;
    cp[t][1] = in.c1[i] + lj;
  }
  const uint64_t* pp = pts + lj;
  int n = 0;
  for (int r = r0; r < r1; ++r, ++n) {
    const uint64_t w = pp[(long long)prow[r] * pstride];
    const long long bo = (long long)idx[r] * bstride;
#pragma unroll
    for (int t = 0; t < TI; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint64_t v = cp[t][c][bo];
        const uint64_t plo = v * w, phi = __umul64hi(v, w);
        const uint64_t s = lo[t][c] + plo;
        hi[t][c] += phi + (s < plo ? 1 : 0);
        lo[t][c] = s;
      }
    }
    if ((n & 15) == 15 && r + 1 < r1) {
#pragma unroll
      for (int t = 0; t < TI; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          lo[t][c] = barrett_lazy(hi[t][c], lo[t][c], p, ratio, shift);
          hi[t][c] = 0;
        }
    }
  }
  const long long is = (long long)limbs * kN, gs = (long long)inputs * is;
#pragma unroll
  for (int t = 0; t < TI; ++t) {
    const int i = i0 + t;
    if (i >= inputs) break;
#pragma unroll
    for (int c = 0; c < 2; ++c)
      res[(long long)c * K * gs + (long long)k * gs + (long long)i * is + lj] = barrett_lazy(hi[t][c], lo[t][c], p, ratio, shift);
  }
}

}  // namespace holmes_mac

using namespace holmes_mac;

// res [2][groups][inputs][limbs][N]; c0s / c1s: host arrays of `inputs` device pointers to the baby bases
// [nb][>= limbs][N] (item stride bstride, limb stride N); pts [groups * nb][>= limbs][N] (row stride pstride).
// ti: 1 / 2 / 4 = TI inputs per thread with all (<= 8) groups, or TI * 100 + TG for a TI x TG tile
// (TG in {1, 2, 4, 8}): 404, 408, 802, 804, 1601, 1602; + 10000 = lazy 128-bit accumulation (mac_multi_lazy):
// 10108, 10204, 10208, 10402, 10404; groups in {1, 2, 4, 8}.  10116 (1 x 16 lazy tile) also takes groups 16 (a
// 1024-row chunk: the baby bases stream once per 16 giants instead of once per 8).
extern "C" int holmes_mac_multi(uint64_t* res, const uint64_t* const* c0s, const uint64_t* const* c1s, long long bstride,
                              int nb, int inputs, const uint64_t* pts, long long pstride, int groups, int ti,
                              int limbs, const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                              void* stream_ptr) {
  if (inputs < 1 || inputs > kMaxInputs) return -1;
  if (nb < 1) return -2;
  if (groups < 1 || groups > 16) return -3;
  if (groups > 8 && !(ti == 10116)) return -3;   // 16 giants (1024-row chunks) need the 1 x 16 lazy tile
  Inputs in;
  for (int i = 0; i < inputs; ++i) { in.c0[i] = c0s[i]; in.c1[i] = c1s[i]; }
  for (int i = inputs; i < kMaxInputs; ++i) { in.c0[i] = c0s[0]; in.c1[i] = c1s[0]; }
  auto st = static_cast<cudaStream_t>(stream_ptr);
  const bool lazy = ti >= 10000;
  if (lazy) ti -= 10000;
  const int TIv = ti < 100 ? ti : ti / 100, TGv = ti < 100 ? 8 : ti % 100;
#define LAUNCH_LAZY(TI, TG)                                                                                       \
  mac_multi_lazy<TI, TG><<<dim3(kN / 256 * ((inputs + TI - 1) / TI), limbs, (groups + TG - 1) / TG), 256, 0, st>>>( \
      res, in, bstride, nb, inputs, groups, pts, pstride, limbs, primes, ratios, ks)
  if (lazy) {
    if (TIv == 1 && TGv == 8) LAUNCH_LAZY(1, 8);
    else if (TIv == 1 && TGv == 16) LAUNCH_LAZY(1, 16);
    else if (TIv == 2 && TGv == 4) LAUNCH_LAZY(2, 4);
    else if (TIv == 2 && TGv == 8) LAUNCH_LAZY(2, 8);
    else if (TIv == 4 && TGv == 2) LAUNCH_LAZY(4, 2);
    else if (TIv == 4 && TGv == 4) LAUNCH_LAZY(4, 4);
    else return -4;
    return static_cast<int>(cudaGetLastError());
  }
#undef LAUNCH_LAZY
#define LAUNCH(TI, TG)                                                                                       \
  mac_multi<TI, TG><<<dim3(kN / 256 * ((inputs + TI - 1) / TI), limbs, (groups + TG - 1) / TG), 256, 0, st>>>( \
      res, in, bstride, nb, inputs, groups, pts, pstride, limbs, primes, ratios, ks)
  if (TIv == 1 && TGv == 8) LAUNCH(1, 8);
  else if (TIv == 2 && TGv == 8) LAUNCH(2, 8);
  else if (TIv == 4 && TGv == 8) LAUNCH(4, 8);
  else if (TIv == 4 && TGv == 4) LAUNCH(4, 4);
  else if (TIv == 8 && TGv == 2) LAUNCH(8, 2);
  else if (TIv == 8 && TGv == 4) LAUNCH(8, 4);
  else if (TIv == 16 && TGv == 1) LAUNCH(16, 1);
  else if (TIv == 16 && TGv == 2) LAUNCH(16, 2);
  else return -4;
#undef LAUNCH
  return static_cast<int>(cudaGetLastError());
}

// res [2][K][inputs][limbs][N]; c0s / c1s: host arrays of `inputs` device pointers to the baby bases [nb][>= limbs][N]
// (item stride bstride); idx [rows] (int32, baby index per row), prow [rows] (int32, plaintext row per row), seg [K + 1]
// (int32 row segment bounds, seg[0] = 0); pts [>= max prow][>= limbs][N] (row stride pstride).  ti = inputs per
// thread (1, 2, 4).
extern "C" int holmes_mac_gather(uint64_t* res, const uint64_t* const* c0s, const uint64_t* const* c1s, long long bstride,
                               const int* idx, const int* prow, const int* seg, int inputs, int K, const uint64_t* pts,
                               long long pstride,
                               int ti, int limbs, const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                               void* stream_ptr) {
  if (inputs < 1 || inputs > kMaxInputs) return -1;
  if (K < 1 || K > 65535) return -3;
  Inputs in;
  for (int i = 0; i < inputs; ++i) { in.c0[i] = c0s[i]; in.c1[i] = c1s[i]; }
  for (int i = inputs; i < kMaxInputs; ++i) { in.c0[i] = c0s[0]; in.c1[i] = c1s[0]; }
  auto st = static_cast<cudaStream_t>(stream_ptr);
#define LAUNCH_G(TI)                                                                                         \
  mac_gather<TI><<<dim3(kN / 256 * ((inputs + TI - 1) / TI), limbs, K), 256, 0, st>>>(                     \
      res, in, bstride, idx, prow, seg, inputs, K, pts, pstride, limbs, primes, ratios, ks)
  if (ti == 1) LAUNCH_G(1);
  else if (ti == 2) LAUNCH_G(2);
  else if (ti == 4) LAUNCH_G(4);
  else return -4;
#undef LAUNCH_G
  return static_cast<int>(cudaGetLastError());
}
