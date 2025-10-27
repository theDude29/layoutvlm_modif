# LayoutVLM modif

Attention : ce dépôt contient des scripts auxiliaires et des expérimentations locales. Pour la documentation complète et officielle du projet LayoutVLM, référez‑vous d'abord au README principal du projet :

https://github.com/sunfanyunn/LayoutVLM

Ce fichier README explique uniquement comment utiliser le petit utilitaire ajouté ici pour extraire des points à éviter à partir d'une image (`scripts/extract_red_path.py`) et comment fournir ces points au solveur via l'option `--proximity_points_file` de `main.py`.

## Dépendances

Le script d'extraction et le pipeline require :
- Python 3.8+
- Pillow
- NumPy

Installez-les si nécessaire :

```bash
pip install pillow numpy
```

(Remarque : le projet principal peut exiger d'autres dépendances comme torch, scipy, shapely, etc. Consultez le README upstream pour la configuration complète.)

## scripts/extract_red_path.py

But: ce script lit une image 2D (fond blanc, chemin/trace rouge) et extrait les coordonnées des pixels considérés "rouges". Il écrit les points dans un fichier texte (une paire `x y` par ligne).

Arguments principaux :
- `--input` (obligatoire) : chemin de l'image (ex: `scripts/path.png`).
- `--output` : fichier de sortie (par défaut `points.txt`).
- `--step` : échantillonnage en pixels (par défaut `1`). Permet de sous-échantillonner pour ne pas écrire tous les pixels.
- `--threshold` : seuil sur le canal rouge (0–255, défaut `200`). Plus élevé = détecte seulement les rouges intenses.
- `--scale_max` : met à l'échelle les coordonnées extraites dans l'intervalle `[0, scale_max]` sur les deux axes (défaut `5.0`). Utile pour produire des points en unités compatibles avec la scène (mètres).
- `--radius` : distance minimale entre points dans l'espace MIS A L'ÉCHELLE (après `--scale_max`). Les points à distance ≤ radius seront supprimés (filtrage glouton). Par défaut `0.0` (pas de filtrage).
- `--origin` : origine pour les coordonnées de sortie. Valeurs possibles : `top-left` (origine en haut‑gauche) ou `bottom-left` (origine en bas‑gauche). Par défaut `bottom-left` (y croît vers le haut).
- `--append` : si présent, les points sont ajoutés au fichier de sortie au lieu d'écraser.

Exemples :

Extraire tous les pixels rouges, mettre à l'échelle dans [0,5], supprimer les points trop proches (radius=0.2):

```bash
python3 scripts/extract_red_path.py --input scripts/my_path.png --output scripts/points.txt --step 1 --threshold 200 --scale_max 5.0 --radius 0.2 --origin bottom-left
```

Créer un fichier en ajoutant aux points existants et en sous-échantillonnant (tous les 2 pixels) :

```bash
python3 scripts/extract_red_path.py --input scripts/my_path.png --output scripts/points.txt --step 2 --append --scale_max 5.0
```

Notes importantes :
- `--scale_max` doit être choisi pour correspondre à l'échelle de votre scène (par exemple une pièce 4×5 m => `scale_max=5.0` fera correspondre l'axe Y à [0,5]).
- `--radius` est interprété après mise à l'échelle (donc en mêmes unités que `scale_max`).
- L'origine `bottom-left` produit des coordonnées avec y croissant vers le haut, compatibles avec la plupart des représentations de scène top‑down.

## Utiliser le fichier de points dans le solveur

Le script principal `main.py` supporte l'option :

```
--proximity_points_file PATH_TO_POINTS_FILE
```

Il lira les lignes `x y` et les transmettra au solveur comme une liste de points `[x, y]` (flottants). Le solveur fournit aussi deux options complémentaires :

- `--proximity_radius` : rayon (en mètres / unités de scène) autour de chaque point pour appliquer une pénalité (valeur par défaut : `1.0`).
- `--proximity_weight` : poids de la pénalité appliquée lorsqu'un objet se trouve à l'intérieur de la zone `proximity_radius` (valeur par défaut : `10.0`).

Exemple : utiliser le fichier généré par `extract_red_path.py` lors de la résolution :

```bash
python3 main.py --scene_json_file ./bedroom_0.json \
  --openai_api_key YOUR_KEY \
  --asset_dir ./data/test_asset_dir \
  --proximity_points_file scripts/points.txt \
  --proximity_radius 0.2 \
  --proximity_weight 20.0
```

Remarques sur l'échelle :
- `extract_red_path.py` peut produire des points déjà mis à l'échelle si vous utilisez `--scale_max`. Dans ce cas, vous pouvez utiliser ces points directement avec `main.py` en choisissant un `--proximity_radius` dans la même unité.
- Si vous fournissez un fichier de points en unités différentes, assurez‑vous que `--proximity_radius` est cohérent.

## Dépannage rapide

- Si `main.py` signale des erreurs de type `NaN` ou des points hors limites, vérifiez :
  - que vos points sont bien dans l'enveloppe du sol (vérifiez `boundary` dans le fichier scene JSON),
  - que `scale_max` correspond à l'échelle de la scène,
  - que les lignes de votre fichier de points sont bien du format `x y` (deux nombres par ligne).

- Pour les dépendances plus larges (PyTorch, SciPy, Shapely), consultez le README principal upstream : https://github.com/sunfanyunn/LayoutVLM


Exemple:
Avec cette image de points à éviter :
![Photo][images/points_rouges.png]
On passe de ça :
![Photo][image/chambre_base.png]
A ça :
![Photo][image/chambre_modif.png]
On a bien laissé le chemin rouge libre dans la config finale !
