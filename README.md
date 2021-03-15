# Projet UE11 - Extraction statistique de l'information

## Installation

### Google Colab

Il est possible d'utiliser les notebooks directement depuis google colab :

`bert.ipynb` : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/al-one-zero/extraction/blob/main/notebooks/bert.ipynb)  
`nnlm.ipynb` : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/al-one-zero/extraction/blob/main/notebooks/nnlm.ipynb)  

### Utilisation en local
1. Cloner le repo
`git clone <url>`
2. Créer un environnement virtuel
```bash
virtualenv venv
./venv/bin/activate
```
3. Installer les dépendances
`pip install -r requirements.txt`

4. Télécharger le modèle de langue `fasttext`  
```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -O data/lid.176.bin
```
(il est possible de télécharger le modèle en version compressé (ftz), modifier l'URL en conséquence)


## Utilisation

Structure du projet :
```
extraction
├── .gitignore
├── data
│   ├── test.txt
│   ├── test_output.txt
│   ├── train.txt
│   └── ...
├── extraction
│   ├── __main__.py
│   ├── preprocessing.py
│   └── training.py
├── mypy.ini
├── notebooks
│   ├── bert.ipynb
│   └── nnlm.ipynb
├── README.md (ce document)
├── rapport.md
├── requirements.txt
└── venv
    └── ...
```

### `./notebooks`

Les notebooks sont le principal outil que l'on utilise pour effectuer nos essais et pour présenter nos resultats.

Les notebooks `bert.ipynb` et `nnlm.ipynb` sont les notebooks à executer pour reproduire nos résultats. Ils contiennent de manière synthétique les différentes étapes de notre chaine de traitement.


### `./extraction`

Le module extraction regroupe les scripts python pouvant être utilisés sous forme de module. Il est surtout utilisé pour le script `preprocessing.py`. `training.py` n'est aujourd'hui pas arrivé à un niveau de maturation suffisant, contrairement aux notebooks nnlm et bert.
