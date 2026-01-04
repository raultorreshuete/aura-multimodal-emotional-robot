# AURA - Robot Social Emocional con IA Multimodal
Desarrollo de un sistema robótico interactivo diseñado para la computación afectiva. El proyecto integra visión artificial, análisis de lenguaje corporal y procesamiento de lenguaje natural para detectar el estado emocional del usuario en tiempo real y proponer actividades de ocio personalizadas.

### 🔧 Hardware e Interfaces (Percepción)
- ***Visión Facial:*** Script dedicado (expresioness_faciales.py) para la captura y análisis de micro-expresiones mediante webcam.
- ***Análisis Corporal:*** Módulo de inferencia (leguaje_corporal.py) que utiliza un modelo pre-entrenado (.pkl) y coordenadas espaciales (coords.csv) para evaluar la postura y el nivel de energía del usuario.
- ***Interfaz de Voz (ASR):*** Módulo de escucha activa (voz.py) que gestiona la captura de audio y su transcripción mediante modelos de IA.
- ***Integración Modular:*** Uso de Stanza y Flask (stanza-flask.py) como middleware para orquestar la comunicación entre los sensores y el cerebro del robot.

### 🏗️ Arquitectura de Software
- ***Gestor de Diálogo (Rasa Core):*** Orquestación de la conversación utilizando historias (stories) que varían según el estado de ánimo detectado, no solo por el texto recibido.
- ***Lógica de Recomendación:*** El script integrador (integrador.py) fusiona los datos de los módulos de visión y voz para tomar decisiones proactivas (ej. sugerir música relajante ante signos de estrés).
- ***Persistencia de Datos:*** Uso de Slots y memoria a largo plazo para recordar gustos y aversiones ("likes/dislikes") del usuario entre sesiones.

### 🚀 Funcionalidades Clave
- ***Detección de Emociones:*** Análisis simultáneo de gestos faciales y postura corporal para inferir estados como alegría, tristeza o estrés.
- ***Recomendación Proactiva:*** El sistema sugiere dinámicamente actividades (ej. poner música relajante si detecta estrés) sin que el usuario lo pida explícitamente.
- ***Modularidad:*** Arquitectura desacoplada donde cada sentido funciona como un microservicio independiente.

### 🛠️ Herramientas y Tecnología
- ***Lenguaje:*** Python 3.x.
- ***Frameworks:*** Rasa (NLP), Flask (Integración), Stanza (Procesamiento), OpenCV/MediaPipe (Visión).
- ***Entorno:*** VS Code y gestión de dependencias con pip.

### ⚠️ Nota de Instalación y Estructura de Archivos
Debido al tamaño de los modelos de lenguaje pre-entrenados y los entornos virtuales, este repositorio contiene los módulos esenciales organizados de la siguiente manera:
- ***RASA:*** Incluye el código fuente de las Custom Actions (actions.py), archivos de configuración (domain.yml, config.yml) y datos de entrenamiento (nlu.yml, stories.yml). No incluye la carpeta models/ ni el entorno virtual.
- ***Lenguaje Corporal:*** Incluye el modelo de clasificación entrenado (body_language.pkl), el dataset de coordenadas de referencia (coords.csv) y los scripts de generación y detección (generarcords.py, leguaje_corporal.py).
- ***Expresiones Faciales:*** Contiene la lógica de visión computacional para la inferencia de emociones (expresioness_faciales.py).
- ***Voz:*** Script de gestión de entrada de audio y reconocimiento (voz.py).
- ***Integración:*** Módulo puente basado en Flask y Stanza (stanza-flask.py, integrador.py) para la comunicación entre componentes.

### 👥 Colaboradores
Proyecto académico desarrollado por Raúl Torres, Miriam Alonso, Borja Hernández, Bartosz Sliwa, Isaac Heredia y Carlos Márquez.
