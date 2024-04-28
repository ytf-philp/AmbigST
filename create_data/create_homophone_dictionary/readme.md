
## Step1: download the G2P model and dictionary from [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/)

## Step2: extract the text data from your dataset

For example:
```
Madame la baronne Pfeffers.
Vous savez aussi bien que moi que de nombreuses molécules innovantes ont malheureusement déçu.
Oh ! parce que maintenant, quand on parle de boire, je m’en vais !
Les questions sanitaires placent l’enfant au cœur de la problématique de l’évolution humaine.
J’ai au moins une satisfaction personnelle : j’ai ému Monsieur Piron.
Avenue du Louvre au numéro trente-huit
```

## Step3: extract the phoneme of all words from text data
```
mfa g2p french_mfa /workspace/fr/source.txt /workspace/fr/phone.txt
nohup mfa g2p spanish_spain_mfa /workspace/AmbigST/homophone/es/source.txt //workspace/AmbigST-de/homophone/es/phone.txt > /workspace/AmbigST/homophone/es/log_dictionary.log 2<&1 &
```

## Step4: cluster the words by using phoneme and create homophone dictionary

python train.py --input_file /workspace/fr/phone.txt --output_file /workspace/AmbigST-de/data_process/fr/phone.csv

We provide the homophone dictionary we construct [Here](create_data/homophone_dictionary)

