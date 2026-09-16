# Décisions de conception

## Choix de `right=False` dans `resample_systematic`

Dans `torch.searchsorted(cumw, points, right=False)` :
- La fonction utilise des semi-ouverts à gauche : $[c_{i-1}, c_i)$.
- Pour un point d'échantillonnage $p$, si $p = c_i$, l'indice retourné est $i$ (et non $i+1$).
- Cela assure la cohérence avec la convention où la première particule est sélectionnée si $p \in [0, w_0]$, évitant qu'une valeur tombant exactement sur le bord supérieur ne déborde prématurément sur la tranche suivante ou hors limites lorsque $p = 1.0$.

## Tests face à l'oracle `particles`

- **Déterminisme et RNG** : `particles.resampling` ne prend pas d'objet générateur en argument explicite et s'appuie sur le runtime NumPy. Les tests d'oracle ne cherchent pas l'égalité bit-à-bit des indices, mais la conformité des lois empiriques.
- **Précision multinomiale vs systématique** : 
  - Pour le multinomial, l'écart d'échantillonnage standard est en $O(k^{-1/2})$, d'où une tolérance statistique fixée à `atol=0.02`.
  - Pour le systématique, le mécanisme de peigne garantit un nombre de copies de la particule $i$ compris entre $\lfloor k w_i \rfloor$ et $\lceil k w_i \rceil$. L'erreur sur chaque fréquence est donc strictement bornée par $1/k$, ce qui autorise un seuil de comparaison strict à `2.0 / k`.