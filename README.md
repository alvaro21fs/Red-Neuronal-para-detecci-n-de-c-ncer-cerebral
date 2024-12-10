<header>
</header>

# Proyecto para detección de tumor cerebral
## Introducción
Se ha utilizado el dataset de Kaggle llamado "Brain Tumor MRI Dataset". Cabe mencionar, que este conjunto de datos se divide en cuatro categorías, una perteneciente a no tumor y otras tres que son distintos tipos de tumores (class_names = ['notumor', 'meningioma', 'glioma', 'pituitary']).

Con las citadas imágenes se han entrenado varios modelos de redes neuronales obteniendo distintos resultados que serán comentados a continuación. Asimismo, se ha utilizado Transfer Learning vía ResNet50 para compararla con la red neuronal propuesta.

## Primera red neuronal propuesta
Se ha propuesto una primera red neuronal basada en redes convolucionales y fully-conected. Esta, se puede encontrar en el cuaderno de Jupyter llamado "Proyecto_Convolusional_Tumor_Cerebral".
Con esta red neuronal se ha obtenido los siguientes resultados, tras entrenarla con un total de 10 epochs (estos resultados pueden ser mejorados con más entrenamiento, pero como prueba de conceptos es más que suficiente lo abordado en este trabajo.

Accuracy of the model on the 21 test batches: 95.73%

Detailed Analysis:
Class: notumor
  True Positives: 270
  False Positives: 10
  False Negatives: 30
  Sensitivity (Recall): 0.90
  Precision: 0.96

Class: meningioma
  True Positives: 285
  False Positives: 30
  False Negatives: 21
  Sensitivity (Recall): 0.93
  Precision: 0.90

Class: glioma
  True Positives: 404
  False Positives: 11
  False Negatives: 1
  Sensitivity (Recall): 1.00
  Precision: 0.97

Class: pituitary
  True Positives: 296
  False Positives: 5
  False Negatives: 4
  Sensitivity (Recall): 0.99
  Precision: 0.98

## Segunda red neuronal propuesta
Para esta red, se ha utilizado Transfer Learning donde se ha utilizado la red residual ResNet50. En ella hemos dejado todas las capas congeladas y hemos añadidos varias capas fully-connected al final para que la salida sea una clasificación entre las cuatro clases citadas.
Con esta red neuronal se ha obtenido los siguientes resultados, tras entrenarla con un total de 20 epochs (estos resultados pueden ser mejorados con más entrenamiento, pero como prueba de conceptos es más que suficiente lo abordado en este trabajo. Cabe mencionar, que hemos realizado el doble de epochs y aun así hemos tartado menos tiempo que con la primera red neuronal.

Accuracy of the model on the 41 test batches: 91.38%

Detailed Analysis:
Class: notumor
  True Positives: 255
  False Positives: 21
  False Negatives: 45
  Sensitivity (Recall): 0.85
  Precision: 0.92

Class: meningioma
  True Positives: 246
  False Positives: 49
  False Negatives: 60
  Sensitivity (Recall): 0.80
  Precision: 0.83

Class: glioma
  True Positives: 405
  False Positives: 31
  False Negatives: 0
  Sensitivity (Recall): 1.00
  Precision: 0.93

Class: pituitary
  True Positives: 292
  False Positives: 12
  False Negatives: 8
  Sensitivity (Recall): 0.97
  Precision: 0.96

Se puede observar que los resultados con esta red, tras realizar más epochs de entrenamiento, no superan a la propuesta, sin embargo, el tiempo de ejecución por epoch de esta último si es muy inferior al de la primera como se había comentado.

<footer>

<!--
  <<< Author notes: Footer >>>
  Add a link to get support, GitHub status page, code of conduct, license link.
-->

---

Get help: [Post in our discussion board](https://github.com/orgs/skills/discussions/categories/github-pages) &bull; [Review the GitHub status page](https://www.githubstatus.com/)

&copy; 2023 GitHub &bull; [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md) &bull; [MIT License](https://gh.io/mit)

</footer>
