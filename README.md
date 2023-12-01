# Lung Cancer Prediction

*Propuesta de solución para la asignatura de Aprendizaje Automático de la Universidad Autónoma del Occidente.*

**Contenido del repositorio**

* **data_exploration.ipybn:** Jupyter Notebook con la exploración inicial de los datos. Se realiza tareas de limpieza y visualización de los datos para preprocesar el data set y comenzar con la codificación del modelo.

## 1. Descripción del problema: 

El diagnóstico médico temprano es un factor clave para el tratamiento de cualquier enfermedad. Posibilita una atención adecuada desde las primeras etapas de la afección, incrementando las posibilidades de recuperación del paciente. 

En este sentido, el cáncer no es la excepción y el diagnóstico temprano beneficia en gran medida al paciente que se atiende a tiempo. Algunos tipos de cáncer se pueden prever con antelación, basándose en algunas características de los pacientes, como los hábitos y sintomatología propia del tipo de enfermedad. Lo anterior, evidencia cierto grado de predisposición de un sujeto de acuerdo con sus hábitos y comorbilidades relacionadas. Así, el cáncer de pulmón,  resulta ser uno de los más comunes en la población. Según las estadísticas del Global Cancer Observatory (Globocan) para el 2020, el cáncer de pulmón fue el segundo cáncer más incidente en el mundo (11,4%), y la principal causa de muerte por cáncer en dicho año, con el 18%. Lo anterior, equivale aproximadamente a 1 de cada 5 muertes por cáncer en la población (Sung et al., 2020). Por tanto, este es un caso de estudio con una relevancia e impacto significativo a nivel global, además de ser un campo con aplicaciones potenciales de tecnologías como la inteligencia artificial para el soporte de los procesos relacionados con el diagnóstico y tratamiento de pacientes. 

Considerando lo anterior, se pretende entrenar un modelo de inteligencia artificial que, en una etapa inicial, identifique si existe predisposición a padecer cáncer de pulmón en un paciente, de acuerdo con la información aportada por él mismo, que permita llegar a dicha impresión diagnóstica. 

### 1.1. Naturaleza supervisada

La capacidad de obtener diagnósticos precisos se beneficia significativamente del aprendizaje supervisado, donde el modelo puede aprender patrones complejos a partir de datos etiquetados. El conjunto de datos a utilizar se compone de 16 características correspondientes a la información de diferentes pacientes recopilada a través de una encuesta. Conforme a dicha información se determinaba el diagnóstico del paciente en una columna objetivo, clasificando aquellos que padecían o no cáncer. 

En ese orden de ideas, los datos de ingesta principales para el modelo son las respuestas correspondientes a cada una de las características de los pacientes, y la variable objetivo o target del modelo son las clases que se definen como el resultado del diagnóstico médico definitivo. Así, el modelo se configura de naturaleza supervisada categórica (IBM, s/f), el cual se vale de un algoritmo para asignar con precisión categorías específicas dentro de un conjunto de datos de prueba y entrenamiento, teniendo presente que se cuentan con los datos de ingesta, definidos como la variable $X$ y el target u objetivo definido como la variable $y$.

### 1.2. Relevancia

El tratamiento temprano del cáncer de pulmón, ha demostrado tener efectos positivos en la supervivencia de un paciente, aumentando su tasa hasta en un 80% (De Lucas et al., 2017). Adicionalmente, se evidencia la necesidad de estrategias de inclusión en la población para el acceso a herramientas de diagnóstico temprano. De esta manera, las aplicaciones del aprendizaje supervisado en herramientas de acceso público para el diagnóstico y alertas tempranas en la predisposición a desarrollar carcinoma pulmonar son fundamentales para las políticas de prevención en salud pública a nivel global. 

### 1.3. Factibilidad

Los datos obtenidos corresponden originalmente al sitio web *online lung cancer prediction system*,sin embargo, actualmente se pueden encontrar en distintos repositorios como [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) , recopilados con el propósito de generar sistemas de información predictivos para estimar una predisposición al cáncer de pulmón mediante la evaluación de diferentes condiciones sintomáticas y hábitos de los pacientes. El diagnóstico de los participantes se confirma por ellos mismos de acuerdo a sus condiciones médicas al momento de diligenciar la encuesta.

En etapas posteriores del modelo se pretende extender el data set en términos de participantes y características para generalizar su aplicación e implementar en herramientas web o aplicaciones móviles para incrementar su accesibilidad. 

## 2. Desarrollo de la propuesta de solución

La propuesta de solución se plantea a partir de los pasos específicos para la exploración de los datos, limpieza y análisis que permitan consolidar un conjunto de datos apropiado para la extracción de información por parte del modelo de aprendizaje automático. Esto, comprende también la selección adecuada de un algoritmo de aprendizaje supervisado para clasificación que logre un nivel de precisión satisfactorio para las predicciones posteriores del modelo. 

Asimismo, se elaborará una arquitectura de la solución que permita entender y replicar el desarrollo del modelo, así como sus resultados esperados. 

### 2.1. Descripción del Fenómeno

El conjunto de datos públicos con los que se trabajará está conformado pro 16 atributos, donde 15 de ellos corresponde a características descriptivas de los pacientes, como los padecimientos y hábitos que presentan, lo que se traduce en las entradas del modelo. El último atributo se refiere a la clase objetivo que es el diagnóstico de cáncer de pulmón en el paciente. 

Según el data set, hábitos como el consumo de alcohol o de tabaco, así como la fatiga o alergias, se relacionan como variables con una potencial predisposición al padecimiento de cáncer de pulmón. El objetivo es utilizar estos atributos para encontrar factores relevantes que determinen el riesgo de desarrollar la enfermedad en cuestión mediante el proceso de aprendizaje automático. 

### 2.2. Obtención de los datos

Los datos recopilados se obtuvieron de un conjunto público implementado en agosto del 2013 por el sitio web online lung cancer prediction system, los cuales se presentan a partir de encuestas de pacientes anónimos que compartieron sus hábitos, síntomas y diagnóstico de manera libre en el sitio. 

Los datos presentan una distribución de género similar, representando una población biológicamente diversa. Asimismo, la población se compone de personas entre 21 y 87 años, con una predominancia entre las edades de 50 a 60 años. 

A continuación se detallan las características del conjunto de datos:

- Gender: representa el género de la paciente. 
- Age: representa la edad del paciente.
- Smoking: Indica si el participante es fumador o no. 
- Yellow fingers: se refiere a si el paciente tiene los dedos amarillos o no. 
- Anxiety: indica si el participante sufre de ansiedad o no. 
- Peer pressure: refleja si el participante percibe presión grupal o no. 
- Chronic desease: señala si el participante presenta alguna enfermedad crónica o no. 
- Fatigue: manifiesta si el participante sufre de fatiga o no. 
- Allergy: indica si el participante sufre de alguna alergia o no.
- Wheezing: hace referencia a si el participante presenta sibilancias o no. 
- Alcohol: el participante señala si consume alcohol o no. 
- Coughing: se refiere a si el paciente sufre de tos o no. 
- Shortness of breath: indica si el participante presenta dificultades para respirar o no. 
- Swallowing difficulty: El participante señala si tiene dificultades para ingerir alimentos o no. 
- Chest pain: se refiere a si el participante presenta dolor en el pecho o no. 
- Lung Cancer: indica si el paciente ha sido diagnosticado con cáncer o no. 

Todas las características del conjunto de datos son nominales, con excepción de la edad que es un dato numérico. Actualmente, se dispone del data set en distintos repositorios de la web como Kaggle y no se cuenta con un registro o intenciones de actualización o expansión de los datos registrados. 

### 2.3. Descripción del problema de Aprendizaje Automático

Se pretende desarrollar un sistema de apoyo a la decisión clínica que pueda diagnosticar el cáncer de pulmón a partir de datos médicos, por medio de técnicas de aprendizaje automático supervisado para entrenar un modelo de clasificación binaria que pueda predecir la predisposición de cáncer de pulmón en el paciente. 

Se compararán diferentes algoritmos de aprendizaje supervisado y se seleccionará el que ofrezca el mejor rendimiento, el cual se medirá a través de métricas como la precisión, el recall, entre otras, con el propósito de evaluar la capacidad del modelo para clasificar correctamente los casos positivos y negativos.  

### 2.4. Visualización y Procesamiento de Datos

Las técnicas de visualización y preprocesamiento de datos que se aplicarán sobre el conjunto de datos permitirán la exploración de las características, la distribución y la relación entre las variables, así como la identificación de anomalías o valores faltantes. Para dicho proceso se pretende utilizar librerías como `Matplotlib` y `Seaborn` de Python, que permiten la generación de gráficos para soportar la descripción y entendimiento de los datos. 

Las gráficas de visualización se pueden consultar en el archivo ***`data_exploration.ipynb`*** incluido en este repositorio. 

**Preprocesamiento:**

En cuanto al preprocesamiento de los datos, se utilizarán las librerías `pandas` y `scikit-learn` de Python, que ofrecen funciones para la manipulación y transformación de los datos. Algunas técnicas de preprocesamiento que se planean utilizar según el caso son:

- Eliminación de los valores faltantes: El conjunto de datos no presenta valores nulos o faltantes. 
- Tratamiento de los valores atípicos para reducir el ruido y la variabilidad de los datos: Los valores atípicos se presentan principalmente en la participación por edad de la población. 
- Codificación de las variables categóricas: El data set está compuesto en su mayoría por datos categóricos que se codifican de forma numérica para el correcto procesamiento del modelo. 
- Normalización o estandarización de las variables numéricas: La única variable numérica es la edad, por lo tanto, no se considera necesaria la estandarización de los datos.  
- Selección o extracción de características para reducir la dimensionalidad: Se plantea utilizar el procesamiento PCA para la extracción de componentes principales con el propósito de mejorar la precisión del modelo. 

# Referencias

Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: A Cancer Journal for Clinicians, 71(3), 209–249. https://doi.org/10.3322/caac.21660

¿Qué es el aprendizaje supervisado? (s/f). Ibm.com. Recuperado el 23 de noviembre de 2023, de https://www.ibm.com/mx-es/topics/supervised-learning. 

De Lucas, M., & de los Ángeles, M. (2017). Valor de los compuestos orgánicos volátiles en aire exhalado en el diagnóstico del cáncer de pulmón. https://docta.ucm.es/entities/publication/94cd91ad-6042-46dc-9b75-0bbdbb16470f.  