# MLLMs as Listeners

Code for the paper ["Are Multimodal Large Language Models Pragmatically Competent Listeners in Simple Reference Resolution Tasks?"](https://aclanthology.org/2025.findings-acl.1236/) (ACL Findings 2025)

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
@inproceedings{junker-etal-2025-multimodal,
    title = "Are Multimodal Large Language Models Pragmatically Competent Listeners in Simple Reference Resolution Tasks?",
    author = "Junker, Simeon  and
      Ali, Manar  and
      Koch, Larissa  and
      Zarrie{\ss}, Sina  and
      Buschmeier, Hendrik",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1236/",
    pages = "24101--24109",
    ISBN = "979-8-89176-256-5",
    abstract = "We investigate the linguistic abilities of multimodal large language models in reference resolution tasks featuring simple yet abstract visual stimuli, such as color patches and color grids. Although the task may not seem challenging for today{'}s language models, being straightforward for human dyads, we consider it to be a highly relevant probe of the pragmatic capabilities of MLLMs. Our results and analyses indeed suggest that basic pragmatic capabilities, such as context-dependent interpretation of color descriptions, still constitute major challenges for state-of-the-art MLLMs."
}
```
