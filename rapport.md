---
title: Rapport
authors: 
 - Oussama Bouzaouit
 - Jean-Marc Fares
date: 15/03/21
git-repo: https://github.com/al-one-zero/extraction
---

Extraction des données
===

0 - Introduction
---
Le problème qui nous est proposé est un problème de traitement de la langue et d'analyse du sentiment d'un corpus de messages extraits du site de microblogging Twitter. On se propose de déterminer le sentiment que décrivent les tweets vis a vis d'une des entreprises suivantes : Google, Apple, Microsoft et Twitter.
On nous propose de discriminer les tweets en 4 catégories : sentiment positif, négatif, neutre et les tweets qui ne sont pas en rapport avec l'entreprise concernées.

Dans ce document, nous décrirons d'abord les données et l'adaptation que l'on en fait pour l'étude, les approches abordées et les résultats que l'on obtient à l'issue de l'approche choisie.

1 - Présentation des données et traitement préalable
---
Les données se présentent dans un fichier texte par ensemble comportant un tweet par ligne. A chaque tweet sont associés un indice, une étiquette de setiment parmis "pos", "neg", "neu" et "irr" respectivement pour designer les sentiments positif, négatif, neutre et sans rapport, et une étiquette désignant l'entreprise concernée par le tweet. Un extrait des données est fourni en Figure 1.

```
(0000,neu,apl) 20 min line @apple store @short pump.
(0001,irr,msf) Nueva tecnología convierte cualquier superficie en una pantalla multitactil. http://t.co/EDibLL5V #Microsoft #omnitouch
(0002,neu,ggl) Some people should not post replies in #Google+ threads. Their posts only continue to weaken their creditbility.
(0003,neg,apl) I know a few others having same issue RT @Joelplane: 9% now on my second full charge of the day. Pissed @Apple
(0004,neg,msf) #Microsoft - We put the ""backwards"" into backwards compatibility. #instantfollowback
(0005,neg,twt) #twitter is sooo trash ritenow with all dezz #highscoolmemories -__-
(0006,neu,apl) RT @jesperordrup: Hi @apple. Household has 4 iphones, 2 ipads, 2 minis, 2 apple tv, AirPorts, Timecapsule +. Whats a usable iPhoto shari ...
(0007,irr,msf) #ALG Culminando formación en #Microsoft M-2778 y en #IndicadoresDeGestión. Gracias a nuestros clientes por su preferencia. www.alg.net.ve
(0008,neu,msf) #Microsoft Community Blogs The 7/365 Review - The Cloud's Impact on Business http://t.co/gPB2MjFW
(0009,irr,twt) Buenas noches a todos #Twitter off
```
_Fig. 1 :_ Extrait du corpus d'entrainement des tweets

Nous importons ce fichier dans un environnement `python` à l'aide de la bibliothèque [`pandas`](http://pandas.pydata.org).

### Extraction des _mentions_ et des _hashtags_ [[source]](https://github.com/al-one-zero/extraction/blob/2b1308c63e731643da9f3b4c6b174716c13e6873/extraction/preprocessing.py#L54)

Un premier traitement que nous réalisons est celui d'extraire les hashtags et les mentions contenues dans chaque tweet. Nous remplacons les mots de ces deux types de lien propre à twitter par leur contenu textuel : "#microsoft" devient "microsoft" et "@apple" devient "apple" par exemple. Nous plaçons ensuite toutes les mentions dans une liste que l'on ajoute à la ligne du tweet concerné, de même pour les mentions.

### Remplacement des émojis [[source]](https://github.com/al-one-zero/extraction/blob/2b1308c63e731643da9f3b4c6b174716c13e6873/extraction/preprocessing.py#L31)

De plus, pour rendre compte de l'information contenue dans les émojis, on décide de les remplacer par leur nom unicode (le nom sous lequel ils sont décrits dans la norme les introduisant).  
Ainsi, l'émoji "🥓" ayant pour texte unicode "\:bacon:" devient le mot "bacon", ou bien "🤙" ayant pour texte  "\:call_me_hand:" devient "call me hand".

### Liens hypertexte [[source]](https://github.com/al-one-zero/extraction/blob/2b1308c63e731643da9f3b4c6b174716c13e6873/extraction/preprocessing.py#L20)

Les liens hypertexte (c'est-à-dire les chaînes de caractères préfixées par "http(s)://bit.ly" - car twitter raccourcit automatiquement les liens avec le raccourcisseur _bit.ly_) sont remplacés par la chaîne "\_LINK_".

## 2 - Méthodologie de prédiction du sentiment

### a) Détection de la langue du tweet [[source]](https://github.com/al-one-zero/extraction/blob/2b1308c63e731643da9f3b4c6b174716c13e6873/extraction/preprocessing.py#L87)

On constate qu'une grande majorité des tweets de la classe "irr" sont formulés dans une langue autre que l'anglais. Forts de ce constat, on se propose d'ajouter une information sur la langue de chaque tweet.  
Pour automatiser le processus, on utilise un modèle préentrainé de la librairie [`fasttext`](https://fasttext.cc/docs/en/language-identification.html). Ce module permet d'identifier 176 langues et est entrainé sur les corpus de texte de Wikipédia, de SETimes et du corpus de traduction collaboratif Tatoeba.  
En pratique, on interroge le modèle pour chaque tweet, et l'on ajoute le bigramme correspondant au tweet - dans une nouvelle colonne, ainsi que la probabilité avec laquelle le tweet est formulé dans la langue citée.

### b) Plongements de texte

Afin de pouvoir présenter le contenu des données à un prédicteur, nous pouvons utiliser des plongements de texte pour transformer le contenu textuel du tweet en une représentation numérique.  
Parmis les options qui s'offrent à nous, on choisit d'uiliser des réseaux d'embedding préentrainés sur des corpus de texte plus imposants que le notre. Pour cette seconde option, on se propose d'essayer les embeddings [nnlm](https://tfhub.dev/google/nnlm-en-dim128/2) et [BERT](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3).

En pratique, l'utilisation des deux types de modèles sont équivalents autant dans la mise en place de la solution que dans les résultats.
Pour produire une représentation vectorielle de nos tweets, il nous faut télécharger le modèle préentrainé depuis internet et le charger à l'aide respectivement des modules `tensorflow_hub` et `tensorflow`.  
La marche à suivre est donnée sur la page de chacun des modèles sur [tfhub.dev](https://tfhub.dev). Une légère subtilité est que BERT n'attend pas une chaîne de caractères en entrée, le tweet doit être tokenisé (découpage de la chaîne de catractères initiale en liste de mots). Le préprocesseur produit pour chaque tweet le triplet de tenseurs entiers suivants : un tenseur des indices des mots utilisés, un tenseur représentant le masque que l'on a sur le tenseur précédent (afin que tous les tweets aient la même longueur, on remplit les tweets les plus courts pour qu'ils aient la même longueur que le plus long), puis un tenseur des débuts des tokens dans le tweet. Le modèle `nnlm` luiu attent directement des chaînes de caractère, donc il n'y a pas de préprocessing à faire en amont.  

### c) Entreprise et langue

Nous incluons l'information sur le nom de l'entreprise que concerne le tweet. Au même titre que pour l'information de langue, il nous faut les transformer en donnée numérique. On choisit simplement de transformer les deux variables en variables catégorielles, on remplace les chaines de caractère par les indices desdites variables, ces indices nous les transformons par la suite en vecteurs one-hot.  

### d) Predicteur

Nous choisissons d'utiliser un réseau de neurones profonds avec quatre couches cachées :
  - une couche cachée pour chacune des colonnes "Language" et "Entreprise" de 4 neuronnes,
  - deux couche cachées communes de 128 et 64 neuronnes placées après la couche de concaténation des trois entrées, suivies chacune d'une couche _dropout_ - elle est responsable de l'introduction d'un bruit, qui permet de réduire l'impact du surapprentissage.  

Nous utilisons `tensorflow.keras` pour implémenter cette méthode d'apprentissage. Une vision schématique de ce réseau de neuronnes est en Figure 2.
![Schematisation du modèle](https://i.imgur.com/ADB8Dfs.png)
_Fig. 2:_ Schématisation du modèle


Nous pouvons enfin formuler la remarque suivante : afin de combatre d'autant plus efficaceement le phénomène de sur apprentissage, on ajoute de la régularisation L1 sur les poids internes et de la régularisation L2 sur les sorties de nos couches cachées.

## 3 - Résultats

Pour entrainer nos modèles, nous avons séparé le jeu initial fourni dans le fichier `train.txt` en trois sous-ensembles : train, validation (à la volée pendant l'entrainement) et test.  

Il est essentiel de remarquer que l'on a effectué les entrainements de nos modèles à l'aide d'une machine exploitant une accélération graphique. Chaque époque demandant environ une minute d'execution. Sur une machine personnelle, cela varie et est de l'ordre de la dizaine de minutes.  

En moyenne, le modèle converge assez rapidement, il lui faut entre 2 et 3 époques pour atteindre les 85% d'acquité sur le jeu de données de validation. On atteint dans les meilleurs cas plus de 90% (régulièrement 93%) de bonnes réponses.
De même sur le jeu de test, sur lequel nous obtennons des résultats comparables.  

Pour ce qui est de l'ensemble des données de vérification, nous obtenons la répartition en sentiments suivante : 
```
neu    490
irr    303
neg    133
pos     74
```

## 4 - Conclusion

