# Hackathon RIIAA 2021 "JusticIA para los desaparecidos"

# Título del proyecto "pendiente"

**Nombre del equipo**
SudoSu

**Integrantes**
* Ernesto Castillo
* Alan 
* Bridget
* Carlos



## Descripción

Obtener información relevante de datos de imágenes de documentos históricos es de enorme importancia. Dada la inviabilidad de realizarlo manualmente, el uso de nuevas tecnologías como técnicas de aprendizaje automático y procesamiento de imágenes y texto permiten intentar resolver este problema de manera automática (o semi-automática). 

SudoSu se centrará en el reto 2 que consiste en proponer y desarrollar un algoritmo para la extracción de textos e identificación de personas, organizaciones, lugares y fechas de un conjunto de dos mil imágenes de fichas y expedientes de personas desaparecidas durante la llamada "guerra sucia". El algoritmo a desarrollar contempla la preparación y preprocesamiento automatizado de las imágenes y la extracción de texto mediante un algoritmo de reconocimiento óptico de caracteres (**OCR**) y la identificación de personas, organizaciones, lugares y fechas a partir de técnicas de procesamiento de lenguaje natural (**NLP**).

Además, será importante contemplar que dichos documentos presentan mala calidad de escaneado, tachaduras, enmendaduras, secciones poco legibles, documentos rotados, etc. Además, la ausencia de un formato universal de la información extraída hace que buscar datos relevantes sea una tarea difícil de automatizar.


### Reto
SudoSu participa en el segundo reto, el cual se compone de las siguientes partes: 
Extracción de texto de imágenes con la mayor calidad posible
Extracción de entidades de interés del texto
Es así que dividiremos este reto en los dos subretos correspondientes.


## Pipeline

El proceidmiento que proponemos se compone de la unión de la propuesta de solución a los dos subretos. Individualmente se plantea como sigue:

**Extracción de texto**

1. Hacer una revisión de la calidad de las imágenes y proponer la implementación de técnicas de preprocesamiento semi-automatizadas dadas las características de las imágenes.
2. Hacer transfer learning utilizando los datos extraídos manualmente y una arquitectura de reconocimiento óptico de caracteres (OCR) como Tesseract 4.0. Después se validará la arquitectura implementada con los datos generados automáticamente evaluando el desempeño del entrenamiento y la inferencia del modelo.

**Identificación de entidades de interés**

1. Hacer un preprocesamiento (limpieza) del texto, para eliminar saltos de línea, signos de puntuación o espacios innecesarios. 
2. Utilizando expresiones regulares extraer información con patrones conocidos sobre el texto (folios, fechas, nombres).
3. Entrenar un modelo tipo Transformer para reconocer nombres, lugares, cargos y organizaciones con las bases de datos entregadas. Con esto se espera encontrar entidades que no se encuentran en la base de datos, pero sí en los documentos.
4. De cada documento se extraerán las entidades de interés que se almacenarán en un archivo con extensión csv.

Al completar la etapa de extracción de texto e identificación de entidades de interés se espera que estas sean representativas para realizar una inferencia con poco sesgo, dado que el conjunto de datos con los que se cuenta es una muestra pequeña del universo de expedientes y fichas existentes.


## Cómo correr el código
