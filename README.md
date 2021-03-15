# Projet UE11 - Extraction statistique de l'information

## Installation

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
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
     -O data/lid.176.ftz
```
(il est possible de télécharger le modèle en version binaire, modifier l'URL en conséquence. A noter que par defaut, `extraction.preprocessing` cherche le fichier binaire plutot que le fichier compressé.)


## Utilisation

Structure du projet :
```
extraction
├── .gitignore
├── data
│   ├── df.pk
│   ├── df_preproc.pk
│   ├── test.txt
│   ├── test_df.pk
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
│   ├── jean-marc.ipynb
│   ├── nnlm.ipynb
│   ├── oussama.ipynb
│   └── preprocessing.py
├── README.md (ce document)
├── rapport.md
├── requirements.txt
└── venv
    └── ...
```

### `./notebooks`

Les notebooks sont le principal outil que l'on utilise pour effectuer nos essais et pour présenter nos resultats.
Les notebooks portant nos noms (`oussama.ipynb` et `jean-marc.ipynb`) sont des notebooks bac à sable dans lesquels sont encore présentes les principales traces de recherche.

Les notebooks `bert.ipynb` et `nnlm.ipynb` sont les notebooks à executer pour reproduire nos résultats. Ils contiennent de manière synthétique les différentes étapes de notre chaine de traitement.


### `./extraction`

Le module extraction regroupe les scripts python pouvant être utilisés sous forme de module. Il est surtout utilisé pour le script `preprocessing.py`. `training.py` n'est aujoutd'hui pas arrivé à un niveau de maturation suffisant, contrairement aux notebooks nnlm et bert.