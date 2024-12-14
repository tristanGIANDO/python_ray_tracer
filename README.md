# Path Tracer : Simulateur de Lumière Réaliste

Ce projet implémente un **path tracer**, une méthode avancée de rendu d'images en 3D qui simule la propagation réaliste de la lumière. Contrairement aux techniques de rendu traditionnelles, le path tracing suit chaque rayon de lumière pour déterminer les couleurs visibles à chaque pixel d'une image. Voici une explication détaillée de ses principales caractéristiques.

---

## 🎯 Concept Fondamental : Le Lancer de Rayons

Le **path tracing** utilise une méthode appelée **lancer de rayons**. Pour chaque pixel de l'image, un rayon est envoyé depuis la caméra dans la scène. Ce rayon peut :
1. **Frapper un objet** : On calcule alors les interactions entre le rayon et l'objet (couleur, lumière, réflexions).
2. **Manquer un objet** : Dans ce cas, la couleur du fond est utilisée, souvent définie par une image HDRI.

### Formule de base pour trouver l'intersection entre un rayon et une sphère :
```math
\text{Rayon : } \mathbf{P}(t) = \mathbf{O} + t \cdot \mathbf{D}
```
- \( \mathbf{O} \) : origine du rayon (position de la caméra).
- \( \mathbf{D} \) : direction du rayon.
- \( t \) : distance à laquelle le rayon rencontre la sphère.

Pour une sphère centrée en \( \mathbf{C} \) avec un rayon \( r \), l'équation est :
```math
\|\mathbf{P}(t) - \mathbf{C}\|^2 = r^2
```
Résoudre cette équation revient à déterminer si le rayon touche la sphère.

---

## 💡 Lumière et Ombres

### Éclairage Direct (Shading Diffus)
Lorsqu'un rayon frappe un objet, on calcule l'intensité lumineuse en fonction de l'orientation de la surface par rapport à une source de lumière :
```math
I = \max(0, \mathbf{N} \cdot \mathbf{L})
```
- \( \mathbf{N} \) : vecteur normal à la surface.
- \( \mathbf{L} \) : direction de la lumière.
- \( I \) : intensité lumineuse reçue.

### Ombres
Pour chaque point éclairé, on vérifie si une autre surface bloque la lumière. On envoie un **rayon d'ombre** vers la source lumineuse pour s'assurer qu'il n'y a pas d'obstacles.

---

## 🔄 Réflexions et Rugosité

### Réflexion Spéculaire
La lumière se réfléchit selon la loi de Snell-Descartes. La direction réfléchie est donnée par :
```math
\mathbf{R} = \mathbf{D} - 2 (\mathbf{N} \cdot \mathbf{D}) \cdot \mathbf{N}
```
- \( \mathbf{R} \) : direction réfléchie.
- \( \mathbf{D} \) : direction du rayon incident.
- \( \mathbf{N} \) : vecteur normal.

### Rugosité (Roughness)
Les surfaces rugueuses dispersent la lumière dans plusieurs directions aléatoires. Cette perturbation est simulée en ajoutant une petite variation à la direction réfléchie :
```math
\mathbf{R}_{perturbed} = (1 - \text{roughness}) \cdot \mathbf{R} + \text{roughness} \cdot \text{aléatoire}
```

---

## 🌅 HDRI : Environnements Réalistes

Une **HDRI (High Dynamic Range Image)** est utilisée pour simuler des environnements lumineux complexes. Lorsqu'un rayon ne touche aucun objet, sa direction est utilisée pour "échantillonner" une couleur depuis l'image HDRI.

### Conversion Direction → Coordonnées UV
1. Convertir la direction 3D du rayon (\(x, y, z\)) en coordonnées sphériques (\(\theta, \phi\)).
   - \( \theta = \arccos(y) \)
   - \( \phi = \arctan2(z, x) \)
2. Convertir (\(\theta, \phi\)) en coordonnées UV pour accéder à l'image HDRI.

---

## 🎲 Échantillonnage Monte Carlo

Pour un rendu encore plus réaliste, plusieurs rayons sont lancés pour chaque pixel, chacun avec une légère variation. La couleur finale du pixel est la moyenne des couleurs retournées par ces rayons :
```math
C_{pixel} = \frac{1}{N} \sum_{i=1}^{N} C_{rayon_i}
```
- \( N \) : nombre de rayons par pixel.
- \( C_{rayon_i} \) : couleur obtenue par un rayon donné.

---

## 🚀 Résultat Final

Le path tracing combine tous ces éléments (intersections, réflexions, ombres, rugosité, HDRI, échantillonnage) pour produire des images photoréalistes, comme si elles étaient prises avec une caméra dans un monde réel.
