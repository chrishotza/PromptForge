# MLX-Serve / Gemma 4 / Xcode continuation probe

This public **standalone Python 3.10+ diagnostic** was prepared for [ddalcu/mlx-serve discussion #678](https://github.com/ddalcu/mlx-serve/discussions/678). It is intentionally kept on a separate documentation branch of PromptForge, **not merged into the main research code**.

## Run locally on the Mac hosting MLX-Serve

Keep MLX-Serve running and check its `/v1/models` list. From the directory containing the script:

```bash
python3 gemma4_xcode_probe.py --model mlx-community/gemma-4-12b-it-4bit
```

If a second local model is installed, you can compare it:

```bash
python3 gemma4_xcode_probe.py --model mlx-community/gemma-4-12b-it-4bit --compare-model EXACT_MODEL_ID_FROM_V1_MODELS
```

The default server is `http://127.0.0.1:11234`. To use another local port, set `--base-url http://127.0.0.1:PORT`. If MLX-Serve requires an API key, set `MLX_SERVE_API_KEY` in your environment; the script does **not** include it in the report.

The script performs **five small synthetic input scenarios in both streaming and non-streaming modes** (10 requests per model): plain text, tools declared, a single tool result, two tool results, and a synthetic Swift file-read result. It compares expected output with actual content, thinking markers, tool calls, protocol completion, and timings.

The report is written incrementally to `gemma4_xcode_probe_report.json`. Please **review it before sharing**, and report MLX-Serve version, model ID, Xcode version, relevant outcome counts, and whether you see repeated `<|channel>thought` markers. Do not post private project source, credentials, raw server logs, or real user data.

**Limitations:** Only sends synthetic requests to a loopback HTTP server; it never accesses real Xcode projects, executes generated code, or uses the public internet. Passing these tests does not prove that a real Xcode agent edit/build works. Version comparisons should hold the model, weights, quantization, thinking settings, and context settings constant.

Created November 2026, from discussion logs. This script has **not** been validated against an actual Apple Silicon MLX-Serve instance by the author.