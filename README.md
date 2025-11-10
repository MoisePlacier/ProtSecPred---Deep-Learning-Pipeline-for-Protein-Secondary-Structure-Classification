Petit projet sympa avec les amis d'ACO ! 



50% des AA ont une structure 2nd de type "autre" 


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