# RBVTQuant

`RBVTQuant` is a separate implementation of the soft-relaxation assignment method
described in `RBVT_soft_relaxation_note.md`.

Design choices:

- Keep the block-wise codebook/scaling backbone from `NCCQuant`.
- Support both plain nearest-codeword quantization (`RTN`) and `RBVT`.
- Use one unified entrypoint for quantization, perplexity evaluation, and `lm-eval`.

Main entrypoint:

```bash
cd RBVTQuant
python main.py \
  --model-path <hf-model-or-local-path> \
  --method rbvt \
  --quantizer nf4 \
  --output-dir ./rbvt_model
```

Run plain RTN with the same backbone:

```bash
cd RBVTQuant
python main.py \
  --model-path <hf-model-or-local-path> \
  --method rtn \
  --quantizer nf4 \
  --output-dir ./rtn_model
```

Runtime notes:

- `HF_TOKEN` and `WANDB_API_KEY` are loaded from `RBVTQuant/.env`.
- `lm-eval` uses the same task presets as the reference source. The default preset is `extended`:
  `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`, `boolq`, `rte`, `openbookqa`, `lambada_openai`.
- `wandb` logging is opt-in with `--use-wandb`.
- Only perplexity and `lm-eval` `acc,none` are logged to `wandb`.
