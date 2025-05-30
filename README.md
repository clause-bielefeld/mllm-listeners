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

To be added
