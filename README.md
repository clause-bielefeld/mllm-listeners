# MLLMs as Listeners

Code for the paper "Are Multimodal Large Language Models Pragmatically Competent Listeners in Simple Reference Resolution Tasks?"
(ACL Findings 2025)

## Reproduction steps

1. clone this repo (with `--recurse-submodules` to also clone submodules)
2. run `process_patch_data.ipynb` and `process_colorgrid_data.ipynb` in `data_prep/`
3. run `predict_grid_with_simple_inputs.py` and `predict_patch_with_simple_inputs.py` with all models
4. run the evaluation notebooks 
    - `process_predictions.ipynb`
    - `process_predictions_simplified.ipynb`
    - `process_predictions_directions.ipynb`
    - `process_predictions_conditions.ipynb`

## Citation

```
@inproceedings{junker-2025-multimodal,
  title = {Are multimodal large language models pragmatically competent listeners in simple reference resolution tasks?},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  author = {Junker, Simeon and Ali, Manar and Koch, Larissa and Zarrie{\ss}, Sina and Buschmeier, Hendrik},
  year = {2025},
  publisher = {Association for Computational Linguistics},
  address = {Vienna, Austria}
  pages = {}
  doi = {10.48550/arXiv.2506.11807}
}
```
