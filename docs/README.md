# Agent Instructions & Conventions

## 1. Coding Philosophy & Tensor Management
* **No Defensive Shape Checks:** Do **not** clutter the forward pass with `if x.size(0) != ...`. Assume the caller is responsible for reshaping. In case you can use assertion, but try to avoid them
* **Explicit Shape Annotations:**
    * **ALWAYS** document tensor shapes in docstrings or inline comments.
    * *Format:* `[Batch, Time, Dim]` or `(B, T, D)`.
    * *Example:* `# (B, 84, 84) -> (B, 512)`
* **Variable Naming:** Use mathematical notation where appropriate (e.g., `mu`, `sigma`, `pi` for policy) but keep it readable.

## 2. Documentation & Math
* **Equation Mapping:** If implementing a paper, reference the specific equation number in the docstring (e.g., "Implements Eq. 4 from [Paper Name]").
* **LaTeX Docstrings:** Use LaTeX syntax for math in docstrings if the logic is complex.
* **ArXiv Links:** When using a known trick (e.g., "Gumbel-Softmax"), add the ArXiv link in the comment.

## 3. Experimentation & Reproducibility
* **Config over Hardcoding:** Never hardcode hyperparameters (learning rate, batch size) inside functions. Always pass them as arguments or a config object (e.g., `OmegaConf`, `argparse`).
* **Headless Plotting:** Do not use `plt.show()`. Always save figures to a file (e.g., `plt.savefig(f"plots/{exp_name}.png")`) because I often run on headless servers/clusters.
* **Seed Control:** When creating environments or main loops, always ensure randomness is seeded.

## 4. Libraries & Environment
* **Environment:** Use `/home/castellanoontiv/miniconda3/envs/rl/bin/python`.
