// Fused CKKS encode stage 2 for EasyFHE (uint64 RNS, N = 2^16, slots = N/2, non-ext).
//
// Kernel A reads stage-1 values, rounds and reduces them for each limb, and
// runs NTT rounds 0..9 in shared memory. Kernel B runs rounds 10..15 in place.
// Rounding follows EasyFHE Encode.cu; NTT root order and reductions follow
// NttImpl.cu. The split follows Slot-Coeff's rhombus_pcmm.cu.
#include <cstdint>
#include <cuda_runtime.h>

namespace holmes_enc {

constexpr int kLogN = 16;
constexpr int kN = 1 << kLogN;
constexpr int kWarp = 32;
constexpr long long kMax64 = 9223372036854775295LL;   // Encode.cu MAX_64BIT_VALUE

__device__ __forceinline__ uint64_t barrett64(uint64_t in, uint64_t prime, uint64_t ratio, uint64_t k) {
  uint64_t hi = __umul64hi(in, ratio);
  hi >>= (k - 64);
  uint64_t res = in - hi * prime;
  if (res >= prime) res -= prime;
  return res;
}
__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t mod) {
  return a >= b ? a - b : a + mod - b;
}
__device__ __forceinline__ uint64_t shoup(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t prime) {
  const uint64_t hi = __umul64hi(b_shoup, a);
  return a * b - hi * prime;
}
__device__ __forceinline__ void butterfly(uint64_t& a, uint64_t& b, uint64_t root, uint64_t root_shoup,
                                          uint64_t prime, uint64_t two_p) {
  const uint64_t product = shoup(b, root, root_shoup, prime);
  if (a >= two_p) a -= two_p;
  b = a + (two_p - product);
  a += product;
}
__device__ __forceinline__ void radix8(uint64_t (&x)[8], uint32_t off, const uint64_t* __restrict__ w,
                                       const uint64_t* __restrict__ ws, uint64_t p, uint64_t tp) {
  {
    const uint64_t r = w[off], rs = ws[off];
    butterfly(x[0], x[4], r, rs, p, tp); butterfly(x[1], x[5], r, rs, p, tp);
    butterfly(x[2], x[6], r, rs, p, tp); butterfly(x[3], x[7], r, rs, p, tp);
  }
  {
    const uint32_t o = 2 * off;
    butterfly(x[0], x[2], w[o], ws[o], p, tp); butterfly(x[1], x[3], w[o], ws[o], p, tp);
    butterfly(x[4], x[6], w[o + 1], ws[o + 1], p, tp); butterfly(x[5], x[7], w[o + 1], ws[o + 1], p, tp);
  }
  {
    const uint32_t o = 4 * off;
    butterfly(x[0], x[1], w[o], ws[o], p, tp); butterfly(x[2], x[3], w[o + 1], ws[o + 1], p, tp);
    butterfly(x[4], x[5], w[o + 2], ws[o + 2], p, tp); butterfly(x[6], x[7], w[o + 3], ws[o + 3], p, tp);
  }
}
template <int NumRounds>
__device__ __forceinline__ void warp_butterfly(uint64_t& first, uint64_t& second, uint32_t& stage, uint32_t lane,
                                               const uint64_t* __restrict__ w, const uint64_t* __restrict__ ws,
                                               uint64_t p, uint64_t tp) {
  butterfly(first, second, w[stage], ws[stage], p, tp);
#pragma unroll
  for (int s = NumRounds - 2; s >= 0; --s) {
    const uint32_t offset = 1u << s;
    const bool lower = (lane & offset) == 0;
    auto ex = lower ? second : first;
    ex = __shfl_xor_sync(0xFFFFFFFF, ex, offset);
    if (lower) second = ex; else first = ex;
    stage <<= 1;
    const uint32_t ri = stage + (lane >> s);
    butterfly(first, second, w[ri], ws[ri], p, tp);
  }
}

// ---- NTT split (rhombus_pcmm.cu) -----------------------------------------------------------------
// Round l (0 <= l < 16) pairs elements n and n + 2^(15 - l) with root roots[2^l + (n >> (16 - l))].
// Kernel A runs rounds 0..9: they stay inside the column n mod 64 (1024 elements, stride 64).
// Kernel B runs rounds 10..15: they stay inside the aligned 64-element block n / 64, one warp per
// block, lane l holding elements l and l + 32.
constexpr int kCols = 4;                 // columns (residues mod 64) per block of kernel A
constexpr int kBlocks = 64 / kCols;      // blocks per plaintext
constexpr int kColThreads = 128;         // threads per column (1024 elements)
constexpr int kColPad = 1024 + 128 + 4;  // padded shared column: pad(m) = m + 2 (m >> 4)
constexpr int kShared = kCols * kColPad * 8;
constexpr int kThreads = kCols * kColThreads;   // 512
__device__ __forceinline__ int pad(int m) { return m + 2 * (m >> 4); }

// Element m of column cg (residue kCols blockIdx.x + 1 + cg) goes to row[64 m + kCols blockIdx.x + cg]:
// the row keeps residue res at index (res - 1) mod 64 (kernel B undoes this).
__device__ __forceinline__ void store_cols(const uint64_t* __restrict__ sh, uint64_t* __restrict__ o) {
  __syncthreads();
  constexpr int kLanes = kCols / 2, kPerRound = kThreads / kLanes;
  const int q = threadIdx.x % kLanes;
#pragma unroll
  for (int r = 0; r < 1024 / kPerRound; ++r) {
    const int m = threadIdx.x / kLanes + kPerRound * r;
    const int pm = pad(m);
    *reinterpret_cast<ulonglong2*>(o + 64 * m + kCols * blockIdx.x + 2 * q) =
        make_ulonglong2(sh[(2 * q) * kColPad + pm], sh[(2 * q + 1) * kColPad + pm]);
  }
}

// Kernel A.  Block (bx, row): residues res = kCols bx + 1 + cg, cg in [0, kCols) -- column cg holds
// coefficients n = 64 m + res - 1 ... careful: store_cols puts column cg at row index kCols bx + cg =
// (res - 1) mod 64 with res = kCols bx + cg + 1, i.e. column cg holds coefficient residue
// r = kCols bx + cg + 1 (mod 64) and (r == 64 -> 0).  Here we simply define column cg = coefficient
// residue c = (kCols bx + cg + 1) & 63 so that kernel B's read pattern (loc0 = (lane + 63) & 63) is the
// identity used by rhombus_pcmm.
// Coefficient n < slots is in[2 n] (real part of slot n), n >= slots is in[2 (n - slots) + 1]
// (Encode.cu, gap = 1).  Thread tau owns m = tau (n < slots) and m = tau + 512 (n >= slots) of all
// kCols columns: for m < 512 the kCols coefficients 64 m + c, c = c0..c0+3, and their n + slots
// partners are the 8 consecutive doubles in[2 (64 m + c0) .. + 8) (one 64-byte span per thread).
// grid (kBlocks, ceil(limbs / lpb), rows): block y handles limbs [y lpb, min(y lpb + lpb, limbs)).  lpb = limbs
// reads the doubles once per plaintext; lpb = 1 gives limbs x more blocks (small row batches still fill the GPU)
// at the price of re-reading the doubles per limb (L2 hits when the batch is small).
__global__ void __launch_bounds__(kThreads, 2) stage2_phase1(
    uint64_t* __restrict__ out, const double* __restrict__ in, long long in_stride, double sf, int limbs, int lpb,
    const uint64_t* __restrict__ primes, const uint64_t* __restrict__ diffs,
    const uint64_t* __restrict__ ratios, const uint64_t* __restrict__ ks,
    const uint64_t* __restrict__ roots, const uint64_t* __restrict__ roots_shoup) {
  extern __shared__ uint64_t sh[];
  const int cg = threadIdx.x / kColThreads, t = threadIdx.x % kColThreads;
  const int tau = threadIdx.x;
  const int row = blockIdx.z;
  const int limb0 = blockIdx.y * lpb, limb1 = (limb0 + lpb < limbs) ? limb0 + lpb : limbs;
  const double* src = in + (long long)row * in_stride;
  // residues of this block's columns: c(cg) = (kCols bx + cg + 1) & 63
  const int cbase = kCols * blockIdx.x + 1;        // c(cg) = (cbase + cg) & 63; only cg = 3 of bx = 15 wraps to 0
  long long re[2][kCols];
  {
    // m = tau: coefficients n = 64 tau + c(cg).  c = cbase + cg for cg < kCols - 1 or bx < 15; the wrapped
    // column (c = 0) belongs to the *next* m block: coefficient 64 tau + 0 is at in[2 (64 tau)].
#pragma unroll
    for (int kk = 0; kk < kCols; ++kk) {
      const int c = (cbase + kk) & 63;
      const int n = 64 * tau + c;                   // < slots
      const double2 v = *reinterpret_cast<const double2*>(src + 2 * n);
      const long long q0 = llround(v.x * sf);       // coefficient n
      const long long q1 = llround(v.y * sf);       // coefficient n + slots  (m = tau + 512)
      re[0][kk] = (q0 < 0) ? (kMax64 + q0) : q0;
      re[1][kk] = (q1 < 0) ? (kMax64 + q1) : q1;
    }
  }
  const int m0 = tau, m1 = tau + 512;
  uint64_t* mine = sh + cg * kColPad;
  for (int limb = limb0; limb < limb1; ++limb) {
    const uint64_t prime = primes[limb], tp = prime << 1;
    const uint64_t ratio = ratios[limb], bk = ks[limb], diff = diffs[limb];
    const uint64_t* W = roots + (size_t)limb * kN;
    const uint64_t* Ws = roots_shoup + (size_t)limb * kN;
    uint64_t x[8];
    if (limb > limb0) __syncthreads();          // sh free (store_cols of the previous limb done)
#pragma unroll
    for (int kk = 0; kk < kCols; ++kk) {
      uint64_t r0 = barrett64((uint64_t)re[0][kk], prime, ratio, bk);
      uint64_t r1 = barrett64((uint64_t)re[1][kk], prime, ratio, bk);
      if (re[0][kk] > (kMax64 >> 1)) r0 = sub_mod(r0, diff, prime);
      if (re[1][kk] > (kMax64 >> 1)) r1 = sub_mod(r1, diff, prime);
      sh[kk * kColPad + pad(m0)] = r0;
      sh[kk * kColPad + pad(m1)] = r1;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad(t + kColThreads * j)];
    // rounds 0..2: m = t + 128 j, roots 1, 2 + (m >> 8), 4 + (m >> 7)
    radix8(x, 1, W, Ws, prime, tp);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad(t + kColThreads * j)] = x[j];
    __syncthreads();
    // rounds 3..5
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad((t & 15) + 16 * j + 128 * (t >> 4))];
    radix8(x, 8 + (t >> 4), W, Ws, prime, tp);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad((t & 15) + 16 * j + 128 * (t >> 4))] = x[j];
    __syncthreads();
    // rounds 6..8
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad((t & 1) + 2 * j + 16 * (t >> 1))];
    radix8(x, 64 + (t >> 1), W, Ws, prime, tp);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad((t & 1) + 2 * j + 16 * (t >> 1))] = x[j];
    __syncthreads();
    // round 9: m = 8 t + j, pairs (2 q, 2 q + 1) with root 512 + 4 t + q
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad(8 * t + j)];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int ri = 512 + 4 * t + q;
      butterfly(x[2 * q], x[2 * q + 1], W[ri], Ws[ri], prime, tp);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad(8 * t + j)] = x[j];
    store_cols(sh, out + ((size_t)row * limbs + limb) * kN);
  }
}

// Kernel B: rounds 10..15 in place.  Block (blk, limb, row) owns positions blk*1024 .. +1024 of one
// polynomial; warp w the 64-element block 16 blk + w.  Lane l reads the elements of residues l and
// l + 32 (stored at index (res - 1) mod 64 by store_cols), runs the six rounds, and writes elements
// 2 l, 2 l + 1 (natural order, canonical residues) over the same 64 words.  All lanes read before the
// first shuffle, so the in-place write is safe.
__global__ void __launch_bounds__(512) stage2_phase2(
    uint64_t* __restrict__ poly, int limbs, const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ roots, const uint64_t* __restrict__ roots_shoup) {
  constexpr int kData = 1024;
  const int blk = blockIdx.x, limb = blockIdx.y, row = blockIdx.z;
  const uint64_t prime = primes[limb], tp = prime << 1;
  const uint64_t* W = roots + (size_t)limb * kN;
  const uint64_t* Ws = roots_shoup + (size_t)limb * kN;
  const uint32_t lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  uint64_t* base = poly + ((size_t)row * limbs + limb) * kN + (size_t)blk * kData + warp * 64;
  uint64_t f = base[(lane + 63) & 63];
  uint64_t s = base[lane + 31];
  uint32_t stage = 1024 + blk * 16 + warp;             // round 10: root 2^10 + (n >> 6)
  warp_butterfly<6>(f, s, stage, lane, W, Ws, prime, tp);
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    if (f >= prime) f -= prime;
    if (s >= prime) s -= prime;
  }
  __syncwarp();
  *reinterpret_cast<ulonglong2*>(base + 2 * lane) = make_ulonglong2(f, s);
}

}  // namespace holmes_enc

// out[rows][limbs][N] (uint64, any contents) <- NTT(encode(in[rows][2*slots] float64, row stride
// in_stride doubles)).  primes/diffs/ratios/ks: per-limb tables (QplusP_map[limbs] etc.), roots /
// roots_shoup: context.power_of_roots [L+P][N].  Returns cudaGetLastError().
// out[rows][limbs][N] (uint64, any contents) <- NTT(encode(in[rows][2*slots] float64, row stride
// in_stride doubles)).  primes/diffs/ratios/ks: per-limb tables (QplusP_map[limbs] etc.), roots /
// roots_shoup: context.power_of_roots [L+P][N].  rows_per_launch > 0 interleaves kernel A and kernel B
// over that many rows at a time, so kernel B re-reads (and overwrites) the intermediate while it is still
// in L2 and the output is written back to DRAM once (0: one launch each).  parts: bit 0 kernel A, bit 1
// kernel B (3 for the encode; the others only time one kernel), bit 2 side-stream pipelining.  limbs_per_block:
// limbs per kernel A block (0 = all).  Returns cudaGetLastError().
extern "C" int holmes_encode_stage2(void* out, const void* in, long long in_stride, int rows, double sf, int limbs,
                                  const void* primes, const void* diffs, const void* ratios, const void* ks,
                                  const void* roots, const void* roots_shoup, void* stream, int rows_per_launch, int parts, int limbs_per_block) {
  cudaStream_t st = (cudaStream_t)stream;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(holmes_enc::stage2_phase1, cudaFuncAttributeMaxDynamicSharedMemorySize, holmes_enc::kShared);
    attr_set = true;
  }
  if (rows <= 0 || limbs <= 0) return 0;
  const int step = rows_per_launch > 0 ? rows_per_launch : rows;
  const int lpb = (limbs_per_block > 0 && limbs_per_block < limbs) ? limbs_per_block : limbs;
  const int ly = (limbs + lpb - 1) / lpb;
  // parts bit 2: kernel A of batch k + 1 runs on a side stream while kernel B of batch k runs on the caller's
  // stream (small per-batch grids alone leave the GPU underfilled); every A is joined into the caller's
  // stream through an event before its B, so the caller sees the whole encode ordered on its stream.
  const bool pipe = (parts & 4) && step < rows;
  static cudaStream_t side = nullptr;
  static cudaEvent_t ev[2], start;
  if (pipe && !side) {
    cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
    for (int i = 0; i < 2; ++i) cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
  }
  if (pipe) {
    cudaEventRecord(start, st);
    cudaStreamWaitEvent(side, start, 0);   // the doubles / tables are ready on the caller's stream
  }
  cudaStream_t sa = pipe ? side : st;
  int k = 0;
  for (int r0 = 0; r0 < rows; r0 += step, ++k) {
    const int nr = (rows - r0 < step) ? rows - r0 : step;
    uint64_t* o = (uint64_t*)out + (size_t)r0 * limbs * holmes_enc::kN;
    const double* i = (const double*)in + (long long)r0 * in_stride;
    if (parts & 1) holmes_enc::stage2_phase1<<<dim3(holmes_enc::kBlocks, ly, nr), holmes_enc::kThreads, holmes_enc::kShared, sa>>>(
        o, i, in_stride, sf, limbs, lpb, (const uint64_t*)primes, (const uint64_t*)diffs,
        (const uint64_t*)ratios, (const uint64_t*)ks, (const uint64_t*)roots, (const uint64_t*)roots_shoup);
    if (pipe) {
      cudaEventRecord(ev[k & 1], sa);
      cudaStreamWaitEvent(st, ev[k & 1], 0);
    }
    if (parts & 2) holmes_enc::stage2_phase2<<<dim3(64, limbs, nr), 512, 0, st>>>(
        o, limbs, (const uint64_t*)primes, (const uint64_t*)roots, (const uint64_t*)roots_shoup);
  }
  return (int)cudaGetLastError();
}
