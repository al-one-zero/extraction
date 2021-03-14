---
title: Rapport
authors: 
 - Oussama Bouzaouit
 - Jean-Marc Fares
date: 15/03/21
---

Extraction des données
===

0 - Introduction
---
Le problème qui nous est proposé est un problème de traitement de la langue et d'analyse du sentiment d'un corpus de messages extraits du site de microblogging Twitter. On se propose de déterminer le sentiment que décrivent les tweets vis a vis d'une des entreprises suivantes : Google, Apple, Microsoft et Twitter.
On nous propose de discriminer les tweets en 4 catégories : sentiment positif, négatif, neutre et les tweets qui ne sont pas en rapport avec l'entreprise concernées.

Dans ce document, nous décrirons d'abord les données et l'adaptation que l'on en fait pour l'étude, les approches abordées et les résultats que l'on obtient a l'issue de l'approche choisie.

1 - Présentation des données et traitement préalable
---
Les données se présentent dans un fichier texte par ensemble comportant un tweet par ligne. A chaque tweet sont associés un indice, une étiquette de setiment parmis 'pos', 'neg', 'neu' et 'irr' respectivement pour designer les sentiments positif, négatif, neutre et sans rapport, et une étiquette désignant l'entreprise concernée par le tweet. Un extrait des données est fourni en Figure 1.

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

Nous importons ce fichier dans un environnement `python` à l'aide de la bibliothèque `pandas`.

### Extraction des _mentions_ et des _hashtags_

Un premier traitement que nous réalisons est celui d'extraire les hashtags et les mentions contenues dans chaque tweet. Nous remplacons les mots de ces deux types de lien propre à twitter par leur contenu textuel : "#microsoft" devient "microsoft" et "@apple" devient "apple" par exemple. Nous plaçons ensuite toutes les mentions dans une liste que l'on ajoute à la ligne du tweet concerné, de même pour les mentions.

### Remplacement des émojis

De plus, pour rendre compte de l'information contenue dans les émojis, on décide de les remplacer par leur nom unicode (le nom sous lequel ils sont décrits dans la norme les introduisant).
Ainsi, l'émoji '🥓' ayant pour texte unicode '\:bacon:' devient le mot 'bacon', ou bien '🤙' ayant pour texte  '\:call_me_hand:' devient 'call me hand'.

### Liens hypertexte

Les liens hypertexte (c'est-à-dire les chaînes de caractères préfixées par 'http(s)://bit.ly' - car twitter raccourcit automatiquement les liens avec le raccourcisseur _bit.ly_) sont remplacés par la chaîne '_LINK_'.

## 2 - Méthodologie de prédiction du sentiment

### Détection de la langue du tweet

### Plongements de texte

### Predicteurs

## 3 - Résultats

## 4 - Conclusion