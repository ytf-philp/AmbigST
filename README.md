# Speech Sense Disambiguation: Tackling Homophone Ambiguity in End-to-End Speech Translation

This is an implementation of  paper *"Speech Sense Disambiguation: Tackling Homophone Ambiguity in End-to-End Speech Translation"* (read the paper [here]()).

## 👀 Overview

We propose **AmbigST** to mitigate speech sense ambiguity in speech translation.

<div align="left">
  <img src="/Figure/main_figure.png" width="70%">
</div>



#### Result on En-to-XX
<div align="left">
  <img src="/Figure/result1.png" width="70%">
</div>

#### Result on XX-to-En
<div align="left">
  <img src="/Figure/result2.png" width="70%">
</div>

#### Download Trained Models

The models are trained based on pytorch.

| Language Pair | Download Link |
|---------------|---------------|
| AmbigST En-De | [Download]()  |
| AmbigST En-Fe | [Download]()  |
| AmbigST En-Es | [Download]()  |
| AmbigST Fr-En | [Download]()  |
| AmbigST Es-En | [Download]()  |
| AmbigST De-En | [Download]()  |


## ⚙️ Setup

```bash
git submodule update --init SpeechUT/fairseq
cd SpeechUT/
pip install --editable fairseq/
pip install sacrebleu==1.5.1
```


#### Download Pretrained Model

Download the pretrained model of [SpeechUT](https://github.com/microsoft/SpeechT5/tree/main/SpeechUT)


#### Data preparation

ST models are fine-tuned with [fairseq speech-to-text](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text) task, so just follow the data preparation instructions [here](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text#data-preparation).
To fine-tune our released models, you should use the same sentecepiece models and dictionaries as ours:

- En-De: [Sentencepiece Model](/data/MuSTC/en_de/spm_unigram10000.model), [Dict](/data/MuSTC/en_de/dict.spm.txt)
- En-Es: [Sentencepiece Model](/data/MuSTC/en_es/spm_unigram10000.model), [Dict](/data/MuSTC/en_es/dict.spm.txt)
- En-Fr: [Sentencepiece Model](/data/MuSTC/en_fr/spm_unigram10000.model), [Dict](/data/MuSTC/en_fr/dict.spm.txt)
- De-En: [Sentencepiece Model](/data/CoVoST/de_en/spm_unigram10000.model), [Dict](/data/CoVoST/de_en/dict.spm.txt)
- Fr-En: [Sentencepiece Model](/data/CoVoST/fr_en/spm_unigram10000.model), [Dict](/data/CoVoST/fr_en/dict.spm.txt)
- Es-En: [Sentencepiece Model](/data/CoVoST/es_en/spm_unigram10000.model), [Dict](/data/CoVoST/es_en/dict.spm.txt)

We provided examples in [`example`](data/example).


### AmbigST Dataset Construction

To finetune the model we released, you can use the dataset we provide
- En-De: [Train](), [Dev]()
- En-Es: [Train](), [Dev]()
- En-Fr: [Train](), [Dev]()
- En-De: [Train](), [Dev]()
- En-Es: [Train](), [Dev]()
- En-Fr: [Train](), [Dev]()

To construction your own Dataset, please refer to the process we provide in [`create_homophone`](create_data/create_homophone_dictionary) and [`annotate_data`](/data/ytf/AmbigST/create_data/annotate_data)



### Fine-tune an encoder-decoder model

```bash
model_path=path/to/your/pre-trained/model
data_dir=dataset/MuSTC/en-${lang}
bash /speechut/scripts/fine_tune/en_de/all.sh $model_path $data_dir 
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
```
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