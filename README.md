# CGDA
Code and data for paper "CGDA : Enhancing Low-resource Event Causality Extraction via Consistency-Guided Data Augmentation"
## 1. Sample Data

Sample subsets from the original training data using different ratios.

```bash
python utils/sample.py --input_file <input_file> --ratios 1 2 3 5 7 10 --output_prefix <output_path> --seed 2025
```

## 2. Generate Synthetic Data
Generate synthetic documents based on the original dataset.

```bash
python utils/generate_data.py --input_path <input_path> --output_path <output_path>
```


## 3. Score Generated Data
Evaluate the quality of generated data using a rule-based scoring mechanism.

```bash
python utils/dis_score.py --original_data <original_data>  --generated_data <generated_data> --output_path <output_path> --alpha 0.5 --beta 0.5 --keep_ratio 0.5
```

## 4. Train Discriminator Model
Train a discriminator model to distinguish between original and generated samples.

```bash
python discriminator_judgment/train.py --config config/train_config.json
```

## 5. Predict with Discriminator Model
Use the trained discriminator to predict scores or labels for new data.

```bash
python discriminator_judgment/predict.py --input_file <input_file> --output_file <output_file> --model_path <model_path> --tokenizer_path <tokenizer_path> --device cuda:0 --max_seq_len 512
```

## 6. Collaborative Filtering with Small and Large Language Models
Combine predictions from small and large language models to filter high-quality samples.

```bash
python utils/collabrative_filter.py --slm <slm_predict.txt> --llm <llm_filter.txt> --out <result.tsv> --low <low_confidence_threshold> --high <high_confidence_threshold>
```

## 7. Validate Downstream Models
Convert the final filtered dataset into formats suitable for downstream causal extraction models (e.g., BERT-CRF or GlobalPointer), then train and evaluate them in the validation/ directory.

#### Supported Formats:
BERT-CRF: Token-level BIO tagging format.
GlobalPointer: Span-based representation with start/end indices.
Refer to the scripts in validation/ for training and evaluation details.

## Notes
All paths assume a Unix-like filesystem.
GPU is recommended for steps involving deep learning models (train.py, predict.py).
Random seeds (e.g., --seed 2025) ensure reproducibility.
For questions or issues, please contact the authors or open an issue in this repository.

## Acknowledgements
This work was supported by resources and inspirations from the following open-source projects.We sincerely thank the authors and contributors of these repositories for their valuable contributions to the NLP and information extraction communities.

### [1. Cause_Event_Extraction](https://github.com/oyarsa/event_extraction/tree/causal-event-extraction)
Weak Reward Model Transforms Generative Models into Robust Causal Event Extraction Systems
### [2. FastIE](https://github.com/xusenlinzy/FastIE/tree/master)
This project provides a unified framework for training and inference of open-source models for text classification, entity extraction, relation extraction, and event extraction, featuring the following:
✨ Support for multiple open-source models for text classification, entity extraction, relation extraction, and event extraction
👑 Support for training and inference with Baidu's UIE model
🚀 A unified training and inference framework
🎯 Integrated adversarial training methods—simple and easy to use
