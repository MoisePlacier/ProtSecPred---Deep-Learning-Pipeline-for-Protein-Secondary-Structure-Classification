Petit projet sympa avec les amis d'ACO ! 



50% des AA ont une structure 2nd de type "autre" 

## définitions 

### structure primaire

La structure primaire, ou séquence, d'une protéine correspond à la succession linéaire des acides aminés (ou résidus).

### structure secondaire 

La structure secondaire décrit le repliement local de la chaîne principale (sans s'interesser à la chaine latérale) d'une protéine. 
Lorsqu'on analyse une structure de protéine, on observe que la majeure partie des acides aminés adoptent deux structures secondaires régulières : les hélices α et les feuillets β.

Il est possible de représenter la structure des protéines en 1 dimension par un enchaînement d'éléments de structure secondaire.

### structure tertiaire 
on définit la structure d'une protéine comme la conformation tridimensionnelle la plus stable et la plus fréquente dans des conditions physiologiques données (température, pH, solvant, etc.). C’est celle observée par cristallographie aux rayons X, RMN ou cryo-EM. Elle décrit les relations dans l'espace des différentes structures secondaires.

## importance de la tâche 

Recherche médicale etc etc... Applications infinies 

Il est bien plus simple de séquencer une protéine (environ 100$) que de déterminer sa structure par cristallographie aux rayons X, RMN ou cryo-EM. En effet, les méthodes "wet lab" nécessitent un temps énorme (généralement 1 ou plusieurs thèses) et des centaines de milliers de dollars... 


## difficulté de prédire la structure tertiare 

Levinthal paradox (1968) : 
Le liens entre deux acides amninés peut prendre en moyenne 3 états stables. Pour un peptide de 101 acides aminées, ça fait 3^100 conformations possibles. Pour identifier la conformation la plus stable en évaluant toutes les possibilités, ça prendrait un temps immense ! 


## Local vs Global

Les hélices alpha sont des structures locales (il suffit de simplement 5 acides aminés pour créer une boucle et observer la liaison H)

les feillets beta ne sont pas des structures locales (une même séquences peut donner une hélice alpha mais aussi se transformer en feuillet beta par interraction avec une autre séquence lointaine de la protéine)

de ce fait : 35% des structures secondaires sont donc déterminées par des long ranges actions 


Certains acides aminés, comme la proline qui a un résidus aromatique cyclique, induisent des coudes dans la structure des protéines => cela casse l'hélice alpha. 

## secondary structure prediction : première génération de méthodes (1957 - 1878)

regarder à l'échelle d'un unique résidu :


1) une première méthode de prédiction de l'hélice :  c'était qu'a chaque fois qu'on voit une proline, on dit que c'est la fin de l'hélice. 

2) extension de cette approche : construction d'un matrice de probabilité. On prend ttes les séquences dont ont connait la structure. Et pour chaque acide aminé, on mesure la fréquence observée dans chaque état (Helice, feuillet, autre)

les performances de ces approches sont de 50 - 55 % d'accuracy. => c'est bien ? bof

## secondary structure prediction : 2 éme génération de méthodes (gorIII 1983 - 1992)

on ne considère plus la probabilitée associée à 1 seul résidus mais la probabilité conditionelle d'un résidus au centre d'une fenètre de K résidus. Pour gérer les premiers et derniers AA, on créer 2 nouveaux pseudos acides aminé fictifs : AA de début et AA de fin que l'on ajoute en début et fin de chaine pour calculer la proba condit des AA aux extrèmes. 

les performances de ces approches sont de 55 - 60 % d'accuracy => bof car l'approche est tjrs local ! 

## limites de ces approches 

Ces approches restent locales, or 35% des structures secondaires sont donc déterminées par des long ranges actions car la formation des feuillets beta n'est pas locale !  => Ces approches classifies très mal les feuillets beta

aussi, la prédiction se fait à l'échelle de l'AA, hors on sait qu'une hélice alpha c'est minimum 5 AA. Donc si on prédit 3 AA consécutifs comme étant une hélice, on fait comment ? on les transforme en feuillet ? on convertit deux autres AA adjacents comme des hélices ? Pas viable



1) Hydrophobicité / Hydrophilie

Influence l’enroulement des hélices et la formation des feuillets β internes.

ARGP820101 Hydrophobicity index (Argos et al., 1982)

2) Volume / Taille de la chaîne latérale

Les grosses chaînes latérales peuvent stéricalement gêner certains motifs secondaires.

Utile pour prédire les boucles vs hélices compactes.

BIGC670101 Residue volume (Bigelow, 1967)
FAUJ880106 STERIMOL maximum width of the side chain (Fauchere et al., 1988)

3) Polarité

Affects la formation des liaisons hydrogène et l’exposition aux solvants.

CHAM820101 Polarizability parameter (Charton-Charton, 1982)
GRAR740102 Polarity (Grantham, 1974)
RADA880108 Mean polarity (Radzicka-Wolfenden, 1988)

4) Charge électrique

Positif, négatif, neutre → influence les interactions électrostatiques locales, souvent dans les hélices α et les feuillets β.
FAUJ880111 Positive charge (Fauchere et al., 1988)
FAUJ880112 Negative charge (Fauchere et al., 1988)

5) Flexibilité / rigidité

Certains AA sont plus flexibles (Glycine), d’autres rigides (Proline), ce qui impacte la formation de boucles ou de coudes.

BHAR880101 Average flexibility indices (Bhaskaran-Ponnuswamy, 1988)

6) Hydrogène accepteur/donneur potentiel

Indique si l’AA peut participer à des liaisons H locales, critique pour stabiliser α-helices et β-feuillets.

CHAM830107 A parameter of charge transfer capability (Charton-Charton, 1983)
FAUJ880109 Number of hydrogen bond donors (Fauchere et al., 1988)








## Multiple sequence alignments
source : publication [protein net](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2932-0)

"Sequence databases for deriving MSAs were created by combining all protein sequences in UniParc [21] with metagenomic sequences from the Joint Genome Institute and filtering to include only sequences available prior to CASP start dates (Table 2). JackHMMER was then used to construct MSAs for every structure by searching the appropriate sequence database. Different MSAs were derived for the same structure if it occurred in multiple ProteinNets. JackHMMER was run with an e-value of 1e-10 (domain and full length) and five iterations. A fixed database size of 1e8 (option -Z) was used to ensure constant evolutionary distance when deriving MSAs (similar to using bit scores). Only perfectly redundant sequences (100% seq. id.) were removed from sequence databases to preserve fine- and coarse-grained sequence variation in resulting MSAs.

In addition to raw MSAs, PSSMs were derived using Easel [24] in a weighted fashion so that similar sequences collectively contributed less to PSSM probabilities than diverse sequences. Henikoff position-based weights (option -p) were used for this purpose."

## comparaison des méthodes 

PSSM = information évolutive positionnelle spécifique.
BLOSUM = moyenne évolutive globale.
Physico-chimie = propriétés locales instantanées.