# AI RESEARCHER PROTOCOL

## 1. Environment & Compute
- **GPU Safety:** Always check `torch.cuda.is_available()` before moving tensors. Default to "cpu" if not found to prevent crashes during your internal testing.
- **Mocking:** If a training run takes >1 minute, do NOT run it yourself. Create a `mock_train.py` or use a `--dry-run` flag with 1 epoch/1 batch to verify the pipeline works.
- **Dependencies:** Do not assume `flash-attn` or specific CUDA kernels are installed unless verified.

## 2. Code Standards (Research Mode)
- **Typing:** Use `jaxtyping` or `torch.Tensor` type hints with shape comments: `def forward(x: Tensor["b s d"]):`.
- **Config:** Never hardcode hyperparameters. Use `hydra`, `argparse`, or a simple `config.py` dataclass.
- **Reproducibility:** ALWAYS set seeds (random, numpy, torch) at the start of any script.
- **Logging:** Use `wandb` or local JSONL logging. Never rely on console printouts for experiment results.

## 3. Interaction & Reasoning
- **Math First:** When asked to implement a paper/algorithm, write the LaTeX equation in the chat *before* writing code to verify understanding.
- **Data Protection:** NEVER try to `read_file` a dataset larger than 5MB. Read the first 5 lines using `head` or python snippet.
- **Debug Protocol:** If loss goes to NaN, do not just "try again." Write a script to check for gradient explosions or dirty data.

## 4. Memory Maintenance
- Update `activeExperiments.md` after every significant result.
- If you change the model architecture, update `researchContext.md`.

## 5. Env to run experiments
- Use by default the conda env 'rl'