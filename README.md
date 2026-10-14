# Holmes

Holmes provides multi-GPU acceleration for encrypted Llama-3-8B prefill and
autoregressive decode using EasyFHE.

This artifact has been anonymized for paper review. Names, repository references,
and local paths have been replaced, so minor path adjustments may be needed.
Upon paper acceptance, we will publish the original repository link to a
verified release.

## Installation

Use Linux x86_64, Python 3.12, and NVIDIA A100 GPUs. Install EasyFHE
following its [public installation instructions](https://jizhuoran.github.io/EasyFHE/#install),
then install Holmes in the same Python environment from the repository root:

```bash
python -m pip install -e .
```

## Running

Download Meta-Llama-3-8B under its model license. Prepare a QuaRot checkpoint
in a separate standard PyTorch environment:

```bash
mkdir -p tools/third_party
git clone --depth 1 https://github.com/spcl/QuaRot.git tools/third_party/QuaRot
python3 -m venv tools/convert/env_rotate
tools/convert/env_rotate/bin/pip install -r tools/convert/requirements-rotate.txt
ROTATE_PYTHON="$PWD/tools/convert/env_rotate/bin/python" \
    ./tools/prepare_quarot_weights.sh /path/to/Meta-Llama-3-8B
```

In the EasyFHE environment, generate the model fixtures:

```bash
export MODEL_DIR="$PWD/assets/weights/Meta-Llama-3-8B-QuaRot"
python bench/make_model_fixtures.py . fixtures/model --layers 32 \
    --model-dir "$MODEL_DIR"
```

The following examples use eight A100 80 GB GPUs with the `a100-high` preset.
Use `--world-size 2` or `4` for fewer GPUs. Provide sufficient host RAM and
shared memory for the worker contexts and fixtures.

Prefill:

```bash
python bench/t2_model.py --preset a100-high --world-size 8 \
    --transport pinned --fixtures fixtures/model --layers 0-31 \
    --model-dir "$MODEL_DIR" --warm-keys --warmup-layer0 --lite \
    --output artifacts/prefill
```

Prefill followed by autoregressive decode using the encrypted KV cache:

```bash
python bench/t2_prefill_decode.py --preset a100-high --world-size 8 \
    --transport pinned --fixtures fixtures/model --model-dir "$MODEL_DIR" \
    --layers 0-31 --decode-tokens 3 --sl --sl-baby 16 \
    --attention-limb-cap 17 --max-plaintext-rows 512 \
    --warm-keys --output artifacts/prefill_decode
```

Run outputs are generated locally. Use `--help` on either runner for more options.

License: GPL-3.0-or-later; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
