# Aplicación Híbrida para Resumir Documentos

| Apellidos y nombres          | Código    |
|------------------------------|-----------|
| Murillo Dominguez, Paul Hans | 20193507J |

## Descripción del Proyecto
Esta aplicación permite generar resúmenes de documentos extensos utilizando enfoques extractivos y abstractivos. Está diseñada para ofrecer resúmenes coherentes y relevantes a través de una interfaz intuitiva desarrollada con Gradio.

---

## Estructura del Proyecto

```
document-summarizer/
├── app.py                     # Interfaz principal con Gradio
├── README.md                  # Documentación del Space
├── requirements.txt           # Dependencias necesarias
├── modules/                   # Carpeta para módulos personalizados
│   ├── __init__.py            # Archivo para convertir la carpeta en un paquete
│   ├── extractive.py          # Implementación de métodos extractivos
│   ├── abstractive.py         # Implementación de métodos abstractivos
│   ├── preprocessing.py       # Preprocesamiento de documentos (PDF, limpieza, OCR)
│   └── utils.py               # Funciones auxiliares (manejo de texto largo, etc.)
├── examples/                  # Ejemplos de entrada y salida
│   ├── example_input.txt      # Entrada de texto de ejemplo
│   └── sample.pdf             # Documento PDF de ejemplo
└── .gitignore                 # Archivos que deben ser ignorados
```

---

## Detalles de los Archivos

### 1. **app.py**

Archivo principal que define la interfaz Gradio para la aplicación. Proporciona las siguientes funcionalidades:

- Permite a los usuarios ingresar texto manualmente o cargar un archivo en formato PDF o TXT.
- Ofrece tres tipos de resúmenes:
  - **Extractivo**: Usa algoritmos basados en TF-IDF, TextRank o BERT para seleccionar las oraciones más importantes del texto.
  - **Abstractivo**: Genera nuevos resúmenes a partir de modelos preentrenados como Pegasus, T5 y BART.
  - **Combinado**: Aplica un resumen extractivo inicial seguido de un refinamiento abstractivo.
- Incluye opciones para ajustar la longitud del resumen y configurar parámetros como el número de haces para los modelos abstractivos.
- Integra botones para procesar archivos cargados y generar los resúmenes seleccionados.

### 2. **modules/abstractive.py**

Módulo responsable del resumen abstractivo. Contiene:

- **Función `load_summarizers`:** Carga modelos abstractivos preentrenados y sus tokenizadores asociados desde Hugging Face. Esto permite utilizar modelos fine-tuneados en la entrega 2 como Pegasus, T5 y BART directamente.
- **Función `abstractive_summary`:** Genera resúmenes para textos ingresados aplicando los modelos cargados. Configura parámetros como la longitud máxima del resumen y el número de haces de búsqueda.

### 3. **modules/extractive.py**

Este módulo implementa varios métodos para realizar resúmenes extractivos:

- **TFIDFSummarizer:** Calcula la relevancia de cada oración utilizando la técnica de TF-IDF y selecciona las oraciones con mayor puntaje.
- **TextRankSummarizer:** Usa una matriz de similitud entre oraciones y aplica el algoritmo de PageRank para identificar las oraciones más significativas.
- **CombinedSummarizer:** Combina los resúmenes generados por TF-IDF y TextRank para mejorar la relevancia.
- **BERTSummarizer:** Usa un modelo preentrenado de BERT para identificar y extraer las partes clave del texto.

### 4. **modules/preprocessing.py**

Proporciona herramientas para limpiar y preparar los datos antes de generar un resumen:

- **Clase `Preprocessor`:**
  - Limpia textos eliminando caracteres especiales, corrigiendo errores de ortografía y ajustando los espacios entre palabras.
  - Divide el texto en oraciones para su posterior procesamiento.
- **Clase `PDFProcessor`:**
  - Convierte archivos PDF en texto utilizable mediante OCR.
  - Permite procesar documentos con múltiples páginas y extraer contenido relevante.
- **Clase `FileHandler`:**
  - Maneja archivos temporales creados durante el procesamiento de texto.
  - Limpia archivos temporales para mantener el entorno organizado.

### 5. **modules/utils.py**

Incluye funciones auxiliares, como:

- **`handle_long_text`:** Divide textos muy largos en segmentos manejables. Cada segmento se procesa por separado y los resúmenes generados se concatenan al final.
- Valida que los textos procesados cumplan con las restricciones de longitud y otros parámetros requeridos por los modelos abstractivos.

### 6. **requirements.txt**

Contiene todas las dependencias necesarias para ejecutar el proyecto, incluyendo:

- `transformers`: Para trabajar con modelos preentrenados.
- `gradio`: Para construir la interfaz de usuario.
- `python-doctr`: Para realizar OCR en documentos PDF.
- `nltk`, `scikit-learn`, `networkx`: Herramientas utilizadas en los algoritmos extractivos.
- `bert-extractive-summarizer`: Implementación preconstruida para resúmenes con BERT.

---

## Instalación

1. Clonar este repositorio:
   ```bash
   git clone https://huggingface.co/spaces/Overglitch/document-summarizer
   cd document-summarizer
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar la aplicación:
   ```bash
   python app.py
   ```
   
4. Acceder a la interfaz de usuario en el navegador web utilizando la URL proporcionada.
   ```
    Local URL: http://localhost:7860
   ```

---

## Uso

1. Ingresar texto o cargar un archivo PDF/TXT.
2. Seleccionar el tipo de resumen deseado:
   - **Extractivo:** Escoge el método (TF-IDF, TextRank, BERT, etc.) y el número de oraciones a incluir.
   - **Abstractivo:** Configura el modelo abstractivo deseado (Pegasus, T5 o BART), la longitud máxima y el número de haces.
   - **Combinado:** Integra enfoques extractivo y abstractivo en un solo resumen.
3. Presionar el botón "Generar Resumen" para obtener el resultado.
4. Usar el botón "Copiar Resumen" para copiar el texto generado.

---

## Métricas y Evaluación

La calidad de los resúmenes puede evaluarse utilizando:

- **ROUGE:** Para comparar el resumen generado con referencias humanas.
---

## Posibles Mejoras

1. Manejo de texto muy extenso puede requerir segmentación adicional aunque ya se está usando `handle_long_text` que está en utils.py y divide el texto en segmentos manejables.
2. La calidad del OCR depende de la legibilidad del archivo PDF.
3. Los modelos abstractivos pueden generar texto incoherente en algunos casos, especialmente con entradas ambiguas.
4. Se pueden agregar más modelos preentrenados y métodos extractivos para mejorar la calidad de los resúmenes.
5. La interfaz de usuario puede ser mejorada con más opciones de personalización y visualización.
6. Se pueden implementar técnicas de mejora de la coherencia y cohesión en los resúmenes generados.
7. Se pueden agregar opciones para resumir múltiples documentos y comparar los resultados.
8. Agregar idiomas adicionales y modelos multilingües para soportar una mayor variedad de textos.


### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Desarrollo de la aplicación con opciones para ambos tipos de resumen  | 3      |
| Evaluación con usuarios y ajustes realizados según feedback           | 3      |
| Documentación y preparación para despliegue                           | 2      |
| **Total**                                                             | **8**  |

**Exposición (12 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Demostración de la aplicación funcionando y sus características       | 4      |
| Discusión sobre el impacto y posibles aplicaciones                    | 4      |
| Calidad general de la presentación y recursos utilizados              | 4      |
| **Total**                                                             | **12** |