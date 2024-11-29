# **Implementación y Comparación de Técnicas de Resumen Extractivo**

---

| Apellidos y nombres          | Código    |
|------------------------------|-----------|
| Murillo Dominguez, Paul Hans | 20193507J |

## **Tabla de Contenidos**

1. [Introducción](#introducción)
2. [Librerías e Inicialización](#librerías-e-inicialización)
3. [Lectura de Documentos](#lectura-de-documentos)
4. [Preprocesamiento de Texto](#preprocesamiento-de-texto)
5. [Implementación de Técnicas de Resumen](#implementación-de-técnicas-de-resumen)
    - [TF-IDF](#tf-idf)
    - [TextRank](#textrank)
    - [Combinación de TF-IDF y TextRank](#combinación-de-tf-idf-y-textrank)
    - [BERT](#bert)
6. [Evaluación con ROUGE](#evaluación-con-rouge)
7. [Resultados](#resultados)
8. [Análisis de Resultados](#análisis-de-resultados)
9. [Conclusiones](#conclusiones)
10. [Referencias](#referencias)

---

## **Introducción**

El objetivo de esta entrega es implementar diferentes técnicas de resumen extractivo y comparar su desempeño utilizando
la métrica ROUGE. Los métodos implementados son:

- **TF-IDF**: Mide la importancia de las palabras en un documento.
- **TextRank**: Algoritmo inspirado en PageRank para ranking de oraciones.
- **Combinación de TF-IDF y TextRank**: Busca aprovechar las fortalezas de ambos métodos.
- **BERT**: Utiliza modelos de lenguaje preentrenados para capturar relaciones semánticas.

---

## **Librerías e Inicialización**

Importamos las librerías necesarias y descargamos los recursos de NLTK.

```python
import pandas as pd
import re
import nltk
import numpy as np
import networkx as nx
import fitz  # PyMuPDF para lectura de PDF
from typing import List
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarizer import Summarizer  # bert-extractive-summarizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios para NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## **Lectura de Documentos**

Creamos la clase `DocumentReader` para leer archivos PDF y extraer su contenido de texto.

```python
class DocumentReader:
    """Clase para leer documentos PDF y extraer texto plano."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_document(self) -> str:
        """Lee y extrae texto de un archivo PDF usando PyMuPDF."""
        text = ""
        try:
            with fitz.open(self.file_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            raise ValueError(f"Error leyendo el archivo PDF: {e}")
        return text
```

---

## **Preprocesamiento de Texto**

La clase `Preprocessor` se encarga de limpiar y preparar el texto para su análisis.

```python
class Preprocessor:
    """Clase para preprocesar texto: limpieza, tokenización y filtrado."""

    def __init__(self, language: str = 'english'):
        self.stopwords = nltk.corpus.stopwords.words(language)
        self.stemmer = SnowballStemmer(language)
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def tokenize_sentences(text: str) -> List[str]:
        """Tokeniza texto en oraciones."""
        return sent_tokenize(text)

    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """Limpia y preprocesa oraciones eliminando ruido."""
        preprocessed = []
        for sentence in sentences:
            # Filtra oraciones con alta densidad de números
            if sum(char.isdigit() for char in sentence) / max(len(sentence), 1) > 0.3:
                continue

            # Conversión a minúsculas y eliminación de caracteres especiales
            sentence = sentence.lower()
            sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)

            # Tokenización y procesamiento de palabras
            words = word_tokenize(sentence)
            words = [
                self.lemmatizer.lemmatize(self.stemmer.stem(word))
                for word in words if word not in self.stopwords
            ]
            if words:  # Excluir oraciones vacías
                preprocessed.append(' '.join(words))
        return preprocessed
```

**Notas sobre el Preprocesamiento:**

- **Stopwords**: Palabras comunes que no aportan significado significativo (e.g., "the", "and", "is").
- **Stemming**: Reducción de palabras a su raíz (e.g., "running" → "run").
- **Lematización**: Conversión de palabras a su forma base (e.g., "better" → "good").

---

## **Implementación de Técnicas de Resumen**

### **TF-IDF**

**Concepto:**

El **TF-IDF** (Term Frequency-Inverse Document Frequency) es una medida que evalúa la relevancia de una palabra en un
documento dentro de un conjunto de documentos.

La fórmula de TF-IDF para una palabra $ t $ en un documento $ d $ es:

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

Donde:

- $ \text{TF}(t, d) $: Frecuencia del término $ t $ en el documento $ d $.
- $ \text{IDF}(t) $: Inverso de la frecuencia de documentos que contienen $ t $:

$$
\text{IDF}(t) = \log\left(\frac{N}{n_t}\right)
$$

- $ N $: Número total de documentos.
- $ n_t $: Número de documentos que contienen el término $ t $.

**Implementación:**

```python
class TFIDFSummarizer:
    """Genera resúmenes usando el modelo TF-IDF."""

    @staticmethod
    def summarize(sentences: List[str], preprocessed_sentences: List[str], num_sentences: int = 1) -> str:
        """Genera un resumen basado en TF-IDF seleccionando las oraciones mejor puntuadas."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        ranked_indices = np.argsort(sentence_scores)[::-1]
        selected = [sentences[i] for i in ranked_indices[:num_sentences]]
        return ' '.join(selected)
```

**Proceso:**

1. Se calcula la matriz TF-IDF de las oraciones preprocesadas.
2. Se suma la puntuación TF-IDF de cada palabra en una oración para obtener la puntuación de la oración.
3. Se ordenan las oraciones por puntuación y se seleccionan las principales.

---

### **TextRank**

**Concepto:**

**TextRank** es un algoritmo basado en grafos que determina la importancia de las oraciones en función de su similitud
con otras oraciones.

**Implementación:**

```python
class TextRankSummarizer:
    """Genera resúmenes usando el algoritmo TextRank."""

    @staticmethod
    def summarize(sentences: List[str], preprocessed_sentences: List[str], num_sentences: int = 1) -> str:
        """Genera un resumen usando el algoritmo de grafos TextRank."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        ranked_indices = sorted(((scores[node], node) for node in nx_graph.nodes), reverse=True)
        selected = [sentences[i] for _, i in ranked_indices[:num_sentences]]
        return ' '.join(selected)
```

**Proceso:**

1. Se calcula la matriz de similitud de coseno entre las oraciones.
2. Se construye un grafo donde los nodos son oraciones y las aristas representan similitud.
3. Se aplica el algoritmo PageRank para obtener la importancia de cada oración.
4. Se seleccionan las oraciones con mayor puntuación.

---

### **Combinación de TF-IDF y TextRank**

**Concepto:**

La idea es combinar las palabras clave obtenidas por TF-IDF y TextRank para seleccionar oraciones que contengan términos
relevantes en ambos métodos.

**Implementación:**

```python
class CombinedSummarizer:
    """Genera resúmenes combinando palabras clave TF-IDF y TextRank."""

    def __init__(self, top_n_keywords: int = 10):
        self.top_n_keywords = top_n_keywords

    def extract_keywords_tfidf(self, preprocessed_sentences: List[str]) -> List[str]:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        tfidf_scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().sum(axis=0))
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_scores[:self.top_n_keywords]]

    def extract_keywords_textrank(self, preprocessed_sentences: List[str]) -> List[str]:
        words = ' '.join(preprocessed_sentences).split()
        co_occurrence_graph = nx.Graph()
        for i in range(len(words) - 1):
            word_pair = (words[i], words[i + 1])
            if co_occurrence_graph.has_edge(*word_pair):
                co_occurrence_graph[word_pair[0]][word_pair[1]]['weight'] += 1
            else:
                co_occurrence_graph.add_edge(word_pair[0], word_pair[1], weight=1)
        ranks = nx.pagerank(co_occurrence_graph, weight='weight')
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_ranks[:self.top_n_keywords]]

    def combined_keywords(self, preprocessed_sentences: List[str]) -> List[str]:
        tfidf_keywords = self.extract_keywords_tfidf(preprocessed_sentences)
        textrank_keywords = self.extract_keywords_textrank(preprocessed_sentences)
        return list(set(tfidf_keywords) & set(textrank_keywords))

    def summarize(self, sentences: List[str], preprocessed_sentences: List[str], num_sentences: int = 1) -> str:
        keywords = self.combined_keywords(preprocessed_sentences)
        sentence_scores = []
        for i, sentence in enumerate(preprocessed_sentences):
            score = sum(1 for word in sentence.split() if word in keywords)
            sentence_scores.append((score, i))
        ranked_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)
        selected = [sentences[i] for _, i in ranked_sentences[:num_sentences]]
        return ' '.join(selected)
```

**Proceso:**

1. Se extraen las palabras clave principales utilizando TF-IDF y TextRank.
2. Se intersectan las palabras clave para obtener las más relevantes en ambos métodos.
3. Se puntúan las oraciones según la cantidad de palabras clave que contienen.
4. Se seleccionan las oraciones con mayor puntuación.

---

### **BERT**

**Concepto:**

**BERT** (Bidirectional Encoder Representations from Transformers) es un modelo de lenguaje preentrenado que captura
relaciones semánticas y contextuales profundas.

**Implementación:**

```python
class BERTSummarizer:
    """Genera resúmenes usando un modelo BERT extractivo preentrenado."""

    def __init__(self):
        self.model = Summarizer()

    def summarize(self, text: str, num_sentences: int = 1) -> str:
        return ''.join(self.model(text, num_sentences=num_sentences))
```

**Proceso:**

1. Se utiliza el modelo BERT extractivo para generar embeddings de las oraciones.
2. Se seleccionan las oraciones más representativas según el modelo.
3. El modelo tiene en cuenta el contexto y la semántica para seleccionar las oraciones.

---

## **Evaluación con ROUGE**

**Concepto:**

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) es un conjunto de métricas para evaluar automáticamente la
calidad de un resumen comparándolo con uno de referencia.

**Implementación:**

```python
def calculate_rouge_scores(reference_summary: str, generated_summary: str) -> dict:
    """Calcula métricas ROUGE entre un resumen de referencia y el generado."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }
```

**Métricas:**

- **ROUGE-1**: Coincidencia de unigrama (palabras individuales).
- **ROUGE-2**: Coincidencia de bigramas (pares de palabras).
- **ROUGE-L**: Longitud de la subsecuencia común más larga.

---

## **Resultados**

### **Generación de Resúmenes**

Se utiliza un documento de ejemplo (Una entrada de cnn_dailymail dataset) y se genera un resumen con cada técnica.

```python
# Parámetros iniciales
file_path = 'resources/f5df9a1942d2e626c5448416a05ffc9ca2d2369c.txt'  # Ruta del archivo
reference_summary = """
    Deputy police commissioner Nick Kaldas is giving evidence at an inquiry...
    """

num_sentences = 1

# Lectura y preprocesamiento
reader = DocumentReader(file_path)
text = reader.read_document()

preprocessor = Preprocessor()
sentences = preprocessor.tokenize_sentences(text)
preprocessed_sentences = preprocessor.preprocess_sentences(sentences)

# Generar resúmenes
tfidf_summarizer = TFIDFSummarizer()
tfidf_summary = tfidf_summarizer.summarize(sentences, preprocessed_sentences, num_sentences)

textrank_summarizer = TextRankSummarizer()
textrank_summary = textrank_summarizer.summarize(sentences, preprocessed_sentences, num_sentences)

combined_summarizer = CombinedSummarizer()
combined_summary = combined_summarizer.summarize(sentences, preprocessed_sentences, num_sentences)

bert_summarizer = BERTSummarizer()
bert_summary = bert_summarizer.summarize(text, num_sentences)
```

### **Resúmenes Generados**

**Resumen TF-IDF:**

> On the ground: Nick Kaldas, pictured with then police minister John Watkins in 2006, warned on his return from Iraq
> that year that 'Australia is very much part of the international community and if something happens in Palestine or
> Iraq, we have to accept that it has an impact over here' In the lead up to the inquiry, Ms Burn has denied falsifying
> evidence in the warrant to Judge Bell to secretly bug police, or to 'use illegal warrants to secretly record
> conversations of my rivals in the police force'.

**Resumen TextRank:**

> Man of integrity: One of the country's most distinguished police officers, NSW deputy commissioner Nick Kaldas (
> pictured) has worked in Iraq, locked up murderers and trained with the FBI, but his stellar career has been dogged by
> an
> illegal bugging operation commandeered in 2000 by his rival for the top job .

**Resumen Combinado:**

> NSW Deputy Police Commissioner Nick Kaldas told a parliamentary inquiry he has been punished for speaking out,
> including being passed over for promotion, while attacking the way the NSW Ombudsman has handled a two-year
> investigation into the bugging scandal that reaches to the highest levels of the NSW Police Force.

**Resumen BERT:**

> A top policeman who was accused of whistle blowing after he had his home and office bugged by colleagues has told how
> he's been left him 'humiliated' by an investigation into the scandal.

---

## **Análisis de Resultados**

### **Evaluación con ROUGE**

Calculamos las métricas ROUGE comparando cada resumen generado con el resumen de referencia.

```python
# Evaluar con ROUGE
tfidf_rouge = calculate_rouge_scores(reference_summary, tfidf_summary)
textrank_rouge = calculate_rouge_scores(reference_summary, textrank_summary)
combined_rouge = calculate_rouge_scores(reference_summary, combined_summary)
bert_rouge = calculate_rouge_scores(reference_summary, bert_summary)
```

### **Tabla de Resultados**

| Modelo    | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----------|---------|---------|---------|
| TF-IDF    | 0.2817  | 0.0284  | 0.1502  |
| TextRank  | 0.3314  | 0.0809  | 0.2057  |
| Combinado | 0.3218  | 0.1279  | 0.2184  |
| BERT      | 0.2420  | 0.0258  | 0.1529  |

**Interpretación:**

- **ROUGE-1**: El modelo **TextRank** obtuvo la puntuación más alta, seguido de cerca por el modelo Combinado.
- **ROUGE-2**: El modelo **Combinado** superó a los demás, indicando una mejor captura de pares de palabras clave.
- **ROUGE-L**: El modelo **Combinado** obtuvo el mejor puntaje, lo que sugiere que capturó mejor las secuencias de
  palabras similares al resumen de referencia.

---

## **Conclusiones**

- **TF-IDF** es efectivo para identificar oraciones con alta relevancia basada en términos frecuentes, pero puede perder
  contexto.
- **TextRank** captura mejor la importancia de las oraciones considerando la similitud con otras oraciones.
- La **Combinación de TF-IDF y TextRank** mejora la selección al aprovechar las fortalezas de ambos métodos, como se
  refleja en las métricas ROUGE.
- **BERT**, aunque potente en comprensión semántica, puede requerir más ajustes para optimizar su desempeño en resúmenes
  extractivos.

---

## **Referencias**

- [Introducción a TF-IDF](https://en.wikipedia.org/wiki/Tf–idf)


- [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)


- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)


- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)


- Rani, U., & Bidhan, K. (2021). Comparative Assessment of Extractive Summarization: TextRank, TF-IDF and LDA. *Journal
  of scientific research*.


- S. Zaware, D. Patadiya, A. Gaikwad, S. Gulhane and A. Thakare, "Text Summarization using TF-IDF and Textrank
  algorithm," _2021 5th International Conference on Trends in Electronics and Informatics (ICOEI)_, Tirunelveli, India,
  2021, pp. 1399-1407, doi: 10.1109/ICOEI51242.2021.9453071.


- Liu, Y. (2019). Fine-tune BERT for Extractive Summarization. arXiv [Cs.CL]. Retrieved from http://arxiv.org/abs/1903.10318


- Xu, Z., & Zhang, J. (2021). Extracting Keywords from Texts based on Word Frequency and Association Features. *Procedia Computer Science*, 187, 77-82. https://doi.org/10.1016/j.procs.2021.04.035
---

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                      | Puntos |
|---------------------------------------------------------------|--------|
| Implementación del enfoque de resumen extractivo              | 3      |
| Utilización de modelos preentrenados para evaluar importancia | 3      |
| Pruebas iniciales con documentos de ejemplo                   | 2      |
| **Total**                                                     | **8**  |

**Exposición (12 puntos)**

| Criterio                                               | Puntos |
|--------------------------------------------------------|--------|
| Explicación del enfoque extractivo y su implementación | 4      |
| Demostración de resultados iniciales                   | 4      |
| Claridad y coherencia en la presentación               | 4      |
| **Total**                                              | **12** |



