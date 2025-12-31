# BERT-Sequence-Labeling

This repostiory integrates [HuggingFaces](https://github.com/huggingface)'s models in an end-to-end pipeline for sequence labeling. [Here](https://huggingface.co/transformers/pretrained_models.html)
is a complete list of the available models.

## Install

```sh
$ git clone https://github.com/avramandrei/BERT-Sequence-Labeling.git
$ cd BERT-Sequence-Labeling
$ conda create -n bert-sl python=3.10
$ conda activate bert-sl
$ pip install -r requirements.txt
```

## Input Format

The files used for training, validation and testing must be in the following format:
- Each line contains the token and the label separated by space
- Each document or sentence is separated by a blank line

The labels can be whatever you want.

```
This O
is O
the O
first O
sentence B-Label1
. I-Label1

This B-Label2
is I-Label2
the O
second O
```

There can be other columns in the file, and the token-label order can be switched. All
that matters is that you use the correct column indices (starting from 0) when calling
the scripts, and that you keep the sentences or documents separated by a blank line.

## Training

To train a model, use the `train.py` script. This will start training a model that will predict the labels of the column specified by the `[predict_column]` argument.

```
python3 train.py [path_train_file] [path_dev_file] [tokens_column] [predict_column] [lang_model_name]


nohup python3 /data01/hujunjie/ECE/event_extraction-master/sequence_labelling/train.py /data01/hujunjie/ECE/Few-shot-data/fincausal/sample_5%.txt /data01/hujunjie/ECE/RLHF/BIEO_Data/fincausal/dev_bieo.txt 0 1 /data01/hujunjie/PLM_MODEL/roberta-large --save_path /data01/hujunjie/ECE/base_model/fincausal/005 > /data01/hujunjie/ECE/base_model/fincausal/base-model-005.txt &


```

## Inference

To predict new values, use the `predict.py` script. This will create a new file by replacing the predicted column of the test file with the predicted values.

```
python3 predict.py [path_test_file] [model_path] [tokens_column] [predict_column] [lang_model_name]

nohup python3 predict.py /data01/hujunjie/ECE/RLHF/BIEO_Data/Scite/test_bieo.txt /data01/hujunjie/ECE/model/Scite/roberta-large-bieo-scite 0 1 /data01/hujunjie/PLM_MODEL/roberta-large --output_path /data01/hujunjie/ECE/SLM_Result/Scite --output_name roberta-large-full-model-predict.txt > /data01/hujunjie/ECE/SLM_Result/Scite/roberta-large-full-model-predict-log.txt &

nohup python3 predict.py /data01/hujunjie/ECE/RLHF/BIEO_Data/Scite/test_bieo.txt /data01/hujunjie/ECE/model/Scite/roberta-large-bieo-scite-001 0 1 /data01/hujunjie/PLM_MODEL/roberta-large --output_path /data01/hujunjie/ECE/SLM_Result/Scite --output_name roberta-large-001-model-predict.txt > /data01/hujunjie/ECE/SLM_Result/Scite/roberta-large-001-model-predict-log.txt &

nohup python3 predict.py /data01/hujunjie/ECE/RLHF/BIEO_Data/Scite/test_bieo.txt /data01/hujunjie/ECE/model/Scite/roberta-large-bieo-scite-002 0 1 /data01/hujunjie/PLM_MODEL/roberta-large --output_path /data01/hujunjie/ECE/SLM_Result/Scite --output_name roberta-large-002-model-predict.txt > /data01/hujunjie/ECE/SLM_Result/Scite/roberta-large-002-model-predict-log.txt &

nohup python3 predict.py /data01/hujunjie/ECE/RLHF/BIEO_Data/Scite/test_bieo.txt /data01/hujunjie/ECE/model/Scite/roberta-large-bieo-scite-005 0 1 /data01/hujunjie/PLM_MODEL/roberta-large --output_path /data01/hujunjie/ECE/SLM_Result/Scite --output_name roberta-large-005-model-predict.txt > /data01/hujunjie/ECE/SLM_Result/Scite/roberta-large-005-model-predict-log.txt &

/data01/hujunjie/ECE/SLM_Result/FCR/roberta-large-full-model-gen-data-predict-cuda.txt

nohup python3 predict.py /data01/hujunjie/ECE/LLM_GEN_DATA/fcr_gen_data_bieo.txt /data01/hujunjie/ECE/model/FCR/roberta-large-bieo-fcr 0 1 /data01/hujunjie/PLM_MODEL/roberta-large --output_path /data01/hujunjie/ECE/SLM_Result/FCR --output_name roberta-large-full-model-gen-data-predict-cuda.txt > /data01/hujunjie/ECE/SLM_Result/FCR/roberta-large-full-model-gen-data-predict-cuda-log.txt &
```

## Results

#### FGCR

See `data/fgcr` for the data and attribution.

| model | macro_f1 |
| --- | --- |
| bert-base-cased | 73.23 |
| roberta-base | 74.1 |
