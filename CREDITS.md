# Credits

## Method

The refusal-direction intervention method is from:

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery,
  Wes Gurnee, and Neel Nanda. "Refusal in Language Models Is Mediated by a
  Single Direction." NeurIPS 2024. arXiv:2406.11717.
- Reference implementation: https://github.com/andyrdt/refusal_direction
  (Apache-2.0)

Blackfrost AI independently implemented the runtime hook, fitted the
MiniMax-M3.1-Flash direction bank, built the evaluation harness, and packaged
this release. This attribution does not imply that the paper authors authored
these artifacts, participated in this release, or endorse it.

## Upstream model

- MiniMaxAI/MiniMax-M3.1-Flash provides the base checkpoint. Model weights are
  not redistributed in this repository and remain subject to their own license.

## Runtime dependencies

- SGLang and vLLM provide the inference runtimes targeted by the hook.
- PyTorch and NumPy provide the tensor/runtime primitives.

See `NOTICE` and `LICENSE` for details.
