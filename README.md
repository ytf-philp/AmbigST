# 

This is an implementation of  paper *""* (read the paper [here]()).

## 👀 Overview

The motivation of our **PromptST** model is to broaden the abstract representation power of the encoder of S2T models.

<div align="left">
  <img src="https://github.com/ytf-philp/PromptST/assets/54100491/13025542-33c2-4f9b-b22e-3758c088f769" width="70%">
</div>



### Result

From En-to-XX
<div align="left">
  <img src="https://github.com/ytf-philp/PromptST/assets/54100491/13025542-33c2-4f9b-b22e-3758c088f769" width="70%">
</div>

From XX-to-En
<div align="left">
  <img src="https://github.com/ytf-philp/PromptST/assets/54100491/13025542-33c2-4f9b-b22e-3758c088f769" width="70%">
</div>

### Download Trained Models

The models are trained based on pytorch.

|      | **Model**|
| AmbigST En-De   | [Download]()|
| AmbigST En-Fe   | [Download]()|
| AmbigST En-Es  | [Download]()|
| AmbigST Fr-En   | [Download]()|
| AmbigST Es-En   | [Download]()|
| AmbigST De-En  | [Download]()|


## Setup

```bash
git submodule update --init SpeechUT/fairseq
cd SpeechUT/
pip install --editable fairseq/
pip install sacrebleu==1.5.1
```


### Download Pretrained Model

Download the pretrained model of [SpeechUT](https://github.com/microsoft/SpeechT5/tree/main/SpeechUT)


### Data preparation

ST models are fine-tuned with [fairseq speech-to-text](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text) task, so just follow the data preparation instructions [here](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text#data-preparation).
To fine-tune our released models, you should use the same sentecepiece models and dictionaries as ours:

- En-De: [sentencepiece_model](/data/MuSTC/en_de/spm_unigram10000.model), [dict](/data/MuSTC/en_de/dict.spm.txt)
- En-Es: [sentencepiece_model](/data/MuSTC/en_es/spm_unigram10000.model), [dict](/data/MuSTC/en_es/dict.spm.txt)
- En-Fr: [sentencepiece_model](/data/MuSTC/en_fr/spm_unigram10000.model), [dict](/data/MuSTC/en_fr/dict.spm.txt)
- De-En: [sentencepiece_model](/data/CoVoST/de_en/spm_unigram10000.model), [dict](/data/CoVoST/de_en/dict.spm.txt)
- Fr-En: [sentencepiece_model](/data/CoVoST/fr_en/spm_unigram10000.model), [dict](/data/CoVoST/fr_en/dict.spm.txt)
- Es-En: [sentencepiece_model](/data/CoVoST/es_en/spm_unigram10000.model), [dict](/data/CoVoST/es_en/dict.spm.txt)

We provided examples in [`example`](data/example).


### AmbigST Dataset Construction

To finetune the model we released, you can use the dataset we provide
- En-De: [train](), [dev]()
- En-Es: [train](), [dev]()
- En-Fr: [train](), [dev]()
- En-De: [train](), [dev]()
- En-Es: [train](), [dev]()
- En-Fr: [train](), [dev]()

To construction your own Dataset, please refer to the process we provide in [`create_homophone`](create_data/create_homophone_dictionary) and [`annotate_data`](/data/ytf/AmbigST/create_data/annotate_data)



### Fine-tune an encoder-decoder model

```bash
model_path=path/to/your/pre-trained/model
data_dir=dataset/MuSTC/en-${lang}
bash bash /speechut/scripts/fine_tune/en_de/all.sh $model_path $data_dir 
```

Please check the folder [`/speechut/scripts/fine_tune`](speechut/scripts/fine_tune) for detailed configuration.

### Decode
You might average several model checkpoints with the best dev accuracy to stablize the performance,
```bash
python fairseq/scripts/average_checkpoints.py --inputs $model_dir/checkpoint.best_acc*.pt --output $model_dir/checkpoint.avgnbest.pt
```
Then decode the model with beam search,
```bash
model_path=path/to/your/fine-tuned/model
data_dir=dataset/MuSTC/en-${lang}
bash speechut/scripts/inference_st.sh $model_path $data_dir ${lang} tst-COMMON



## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) and [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## Reference

If you find our work is useful in your research, please cite the following paper:

```
```

### Contact Information

For help or issues using AmbigST models, please submit a GitHub issue.

For other communications related to AmbigST, please contact Tengfei Yu (`921692739@qq.com`).