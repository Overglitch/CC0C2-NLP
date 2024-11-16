# Embeddings Gaussianos y RNNs para Modelado de Incertidumbre en Secuencias de Texto


| Apellidos y nombres | Código |
|---------------------|--------|
| Murillo Dominguez, Paul Hans | 20193507J |

## Introducción

Implementación en PyTorch de una RNN que utiliza gaussian
embeddings para generar texto de manera probabilística, permitiendo la captura de múltiples
posibles continuaciones de una secuencia.

Combina gaussian embeddings con RNNs para capturar la incertidumbre y
variabilidad en la generación de secuencias de texto, mejorando la robustez del modelo.

## Tabla de Contenidos

- [Requisitos](#requisitos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Preprocesamiento de Datos](#preprocesamiento-de-datos)
- [Manejo de Secuencias y Padding](#manejo-de-secuencias-y-padding)
- [Embeddings Gaussianos](#embeddings-gaussianos)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Función de Pérdida y Optimización](#función-de-pérdida-y-optimización)
- [Entrenamiento](#entrenamiento)
- [Generación de Texto](#generación-de-texto)
- [Ejemplo de Uso](#ejemplo-de-uso)
- [Referencias](#referencias)

## Requisitos

- Python 3.x
- PyTorch
- NLTK
- NumPy
- Re (Expresiones Regulares)
- [nltk data](https://www.nltk.org/data.html) (Punkt Tokenizer Models)

Instalar los paquetes necesarios:

```bash
pip install torch nltk numpy
```

Descargar los datos de NLTK necesarios:

```python
import nltk
nltk.download('punkt')
```

## Estructura del Proyecto

- `Vocabulary`: Clase para manejar el vocabulario y las conversiones entre palabras e índices.
- `TextDataset`: Clase para cargar y preprocesar el conjunto de datos de texto.
- `GaussianEmbedding`: Implementación de embeddings gaussianos con PyTorch.
- `LSTMModel`: Modelo que combina embeddings gaussianos con un LSTM.
- `Trainer`: Clase para entrenar el modelo y generar texto.

## Preprocesamiento de Datos

El preprocesamiento es fundamental para preparar los datos para el entrenamiento.

### Limpieza de Texto

El texto se limpia para eliminar caracteres y patrones no deseados:

```python
def clean_text(self, text):
    text = re.sub(r'\([^)]*\)', '', text)  # Eliminar texto dentro de paréntesis
    text = re.sub(r'=', '', text)          # Eliminar signos de igual
    text = re.sub(r'<unk>', '', text)      # Eliminar tokens desconocidos
    text = re.sub(r'-{2,}', ' ', text)     # Reemplazar múltiples guiones por un espacio
    text = re.sub(r'\.{2,}', '.', text)    # Reemplazar múltiples puntos por uno solo
    text = re.sub(r"[^a-zA-Z0-9\s\.\']", '', text)  # Eliminar caracteres no deseados
    text = re.sub(r'\s+', ' ', text)       # Reemplazar múltiples espacios por uno solo
    text = text.strip()
    return text
```

### Tokenización

Se utiliza NLTK para tokenizar el texto en oraciones y palabras:

```python
sentences = sent_tokenize(text)
tokenized_sentences = [word_tokenize(sent) for sent in sentences]
```

### Construcción del Vocabulario

Se crea un vocabulario que asigna índices a palabras y viceversa:

```python
self.vocab = Vocabulary()
for sentence in tokenized_sentences:
    self.vocab.add_sentence(sentence)
```

## Manejo de Secuencias y Padding

En tareas de modelado de lenguaje, es común trabajar con secuencias de longitud variable. Para facilitar el procesamiento en lotes (batching) y asegurar que todas las secuencias tengan la misma longitud, utilizamos **padding**. El padding consiste en añadir tokens especiales a las secuencias más cortas para igualarlas en longitud con las más largas.

### Token de Padding (`"<PAD>"`)

- **Propósito**: El token `"<PAD>"` se utiliza para rellenar (pad) las secuencias cortas y mantener una longitud uniforme en todos los lotes.
- **Asignación de Índice**: En la clase `Vocabulary`, el token `"<PAD>"` se asigna al índice `0`.

```python
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0}  # Token de padding
        self.idx2word = {0: "<PAD>"}
        self.idx = 1  # Iniciar índice desde 1
    ...
```

### Creación de Secuencias con Padding

Al crear secuencias de entrada y salida para el modelo, se realiza el padding de las secuencias más cortas:

```python
def create_sequences(self):
    inputs = []
    targets = []
    for sentence in self.data:
        if len(sentence) < 2:
            continue
        indices = self.vocab.sentence_to_indices(sentence)
        for i in range(1, len(indices)):
            seq = indices[max(0, i - self.seq_length):i]
            seq = [0] * (self.seq_length - len(seq)) + seq  # Padding con ceros (índice de "<PAD>")
            inputs.append(seq)
            targets.append(indices[i])
```

- **Proceso**:
  - Para cada palabra en la oración, se crea una secuencia de longitud fija `seq_length`.
  - Si la secuencia es más corta que `seq_length`, se rellena con ceros (el índice de `"<PAD>"`).
  - Esto asegura que todas las secuencias de entrada tengan la misma longitud.

### Ignorar el Padding en la Función de Pérdida

Al calcular la pérdida, es importante no penalizar al modelo por predecir el token de padding. Por ello, en la función de pérdida, se especifica `ignore_index=0` para que el token de padding no contribuya a la pérdida.

```python
self.criterion = nn.CrossEntropyLoss(ignore_index=0)
```

- **Nota**: Esto es crucial para que el modelo no aprenda a predecir `"<PAD>"` como salida válida.

### Uso del Padding en el Modelo

En el modelo, el índice de padding se pasa a los embeddings para que estos sepan qué entradas deben tratarse como padding.

```python
self.embedding = GaussianEmbedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                   padding_idx=padding_idx)
```

- **Parámetro `padding_idx`**: Le indica a la capa de embedding que el índice `padding_idx` corresponde al token de padding y que debe inicializarse a ceros.

## Embeddings Gaussianos

### Concepto

En lugar de representar palabras como vectores determinísticos, los embeddings gaussianos representan cada palabra \[ w \] como una distribución gaussiana multivariante \[ \mathcal{N}(\boldsymbol{\mu}_w, \boldsymbol{\Sigma}_w) \]. Esto permite capturar la incertidumbre y variabilidad semántica.

### Implementación

En el código, los embeddings gaussianos se implementan mediante dos capas de embedding: una para la media y otra para la log-varianza.

```python
self.mean = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
self.log_var = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
```

- **Inicialización**: Los pesos de las capas de embedding se inicializan uniformemente.
- **Padding**: Al especificar `padding_idx`, los embeddings correspondientes al token de padding no se actualizan durante el entrenamiento y permanecen en cero.

### Reparametrización

Para muestrear de la distribución gaussiana y permitir el flujo de gradientes, utilizamos el truco de reparametrización:

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}
$$

Donde:

- \[\boldsymbol{\mu}\] es la media,
- \[\boldsymbol{\sigma} = \exp\left(0.5 \times \log \boldsymbol{\sigma}^2\right)\] es la desviación estándar,
- \[\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})\] es ruido gaussiano,
- \[\odot\] denota multiplicación elemento a elemento.

### Cálculo de la Pérdida KL

La divergencia KL entre la distribución gaussiana de los embeddings y una distribución normal estándar se calcula como:

$$
\text{KL}(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \parallel \mathcal{N}(0, \mathbf{I})) = -\frac{1}{2} \sum_{i=1}^d \left(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2\right)
$$

Implementación en el código:

```python
def kl_loss(self, mean, log_var):
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=2)
    return kl.mean()
```

## Arquitectura del Modelo

El modelo combina los embeddings gaussianos con un LSTM para capturar tanto la incertidumbre semántica como las dependencias secuenciales.

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx, dropout_p):
        super(LSTMModel, self).__init__()
        self.embedding = GaussianEmbedding(...)
        self.lstm = nn.LSTM(...)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, vocab_size)
```

### Flujo de Datos

1. **Embeddings Gaussianos**: Cada palabra se convierte en una distribución gaussiana y se muestrea para obtener \( \mathbf{z} \).

2. **LSTM**: Las secuencias de embeddings muestreados se pasan al LSTM.

3. **Dropout**: Se aplica dropout para regularización.

4. **Capa Fully Connected**: La salida del LSTM se proyecta al espacio del vocabulario para predecir la siguiente palabra.

```python
def forward(self, x, hidden):
    x, mean, log_var = self.embedding(x)
    x, hidden = self.lstm(x, hidden)
    x = self.dropout(x)
    x = self.fc(x[:, -1, :])  # Usar la salida del último paso temporal
    return x, hidden, mean, log_var
```

## Función de Pérdida y Optimización

La función de pérdida combina la pérdida de entropía cruzada (NLLLoss) y la pérdida de divergencia KL:

$$
\text{Pérdida Total} = \text{Pérdida NLL} + \beta \times \text{Pérdida KL}
$$

Donde \( \beta \) es un hiperparámetro que controla la contribución de la pérdida KL.

```python
nll_loss = self.criterion(outputs, targets)
kl_loss = self.model.embedding.kl_loss(mean, log_var)
loss = nll_loss + self.kl_weight * kl_loss
```

Se utiliza el optimizador Adam para actualizar los pesos:

```python
self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

Se aplica clipping de gradiente para evitar explosión de gradientes:

```python
nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
```

## Entrenamiento

El entrenamiento se realiza en múltiples épocas, iterando sobre el conjunto de datos y actualizando los pesos del modelo.

```python
for epoch in range(1, epochs + 1):
    for inputs, targets in self.train_loader:
        # Inicialización del estado oculto
        hidden = self.model.init_hidden(batch_size)
        # Forward y cálculo de pérdidas
        outputs, hidden, mean, log_var = self.model(inputs, hidden)
        # Backpropagation y optimización
        loss.backward()
        self.optimizer.step()
```

Se imprime la pérdida promedio y la perplejidad después de cada época, así como un ejemplo de texto generado.

## Generación de Texto

El modelo genera texto muestreando palabras secuencialmente, utilizando las distribuciones aprendidas.

### Proceso de Generación

1. **Inicialización**: Se proporciona un texto inicial y se inicializa el estado oculto del LSTM.

2. **Muestreo Iterativo**:

   - Se toman las últimas \( n \) palabras como entrada.
   - Se obtiene la distribución de probabilidad sobre el vocabulario.
   - Se selecciona la siguiente palabra muestreando de las \( k \) palabras más probables (estrategia *top-k*).

3. **Actualización**: La palabra generada se añade al texto y el proceso se repite hasta alcanzar la longitud deseada.

```python
def generate_text(self, init_text, length, top_k=5):
    self.model.eval()
    words = word_tokenize(init_text)
    state_h, state_c = self.model.init_hidden(1)
    for _ in range(length):
        # Preparación de la entrada
        input_words = words[-(self.seq_length - 1):]
        indices = [self.vocab.word_to_index(w) for w in input_words]
        if len(indices) < self.seq_length - 1:
            indices = [0] * (self.seq_length - 1 - len(indices)) + indices  # Padding si es necesario
        x = torch.tensor([indices], dtype=torch.long)
        # Generación de la siguiente palabra
        with torch.no_grad():
            output, (state_h, state_c), mean, log_var = self.model(x, (state_h, state_c))
            probs = F.softmax(output, dim=1).data
            top_probs, top_ix = probs.topk(top_k)
            word_idx = np.random.choice(top_ix.cpu().numpy().squeeze(), p=top_probs.cpu().numpy().squeeze() / top_probs.sum())
            word = self.vocab.index_to_word(word_idx)
            words.append(word)
    return ' '.join(words)
```

## Ejemplo de Uso

### Configuración y Entrenamiento

```python
# Parámetros
seq_length = 6
num_sentences = 5000
train_filepath = 'ruta/al/archivo_de_texto.txt'

# Crear el conjunto de datos
train_dataset = TextDataset(train_filepath, seq_length=seq_length, num_sentences=num_sentences)

# Parámetros del modelo
vocab_size = len(train_dataset.vocab)
embedding_dim = 100
hidden_dim = 128
padding_idx = train_dataset.vocab.word_to_index("<PAD>")
dropout_p = 0.1
batch_size = 200
learning_rate = 0.001
num_epochs = 10

# Instanciar el modelo y el entrenador
model = LSTMModel(vocab_size=vocab_size, embedding_dim=embedding_dim,
                  hidden_dim=hidden_dim, padding_idx=padding_idx, dropout_p=dropout_p)
trainer = Trainer(model=model, train_dataset=train_dataset, batch_size=batch_size,
                  lr=learning_rate, kl_weight=0.1)

# Entrenar el modelo
trainer.train(num_epochs)
```

```bash
Número de secuencias de entrada: 107740
Epoch 1, Loss: 7.4279, Perplexity: 1682.2655
Text generated after epoch 1:
The . the the the and the . of . to . the to . the of . the of . and the and the the . of the and the . and the of . of . . . of to the . the . of of . the and

Epoch 2, Loss: 7.1032, Perplexity: 1215.8862
Text generated after epoch 2:
The in . . to of the and and the the of of the the the the . of the . the in the . . to of the the . the . to the in the the the the . in the of the and of the . to in

Epoch 3, Loss: 7.0763, Perplexity: 1183.5990
Text generated after epoch 3:
The of the the the of . of of of the . . . the the in . and to of to and and the the . a to and . the the to the of of the and . to of the the and . the of the . and

Epoch 4, Loss: 7.0363, Perplexity: 1137.1417
Text generated after epoch 4:
The a . of and the the the of and a . . . the . in . of and . a the and . the the the the a . . the the a . of the in in the a of . a the the and the and a

Epoch 5, Loss: 6.9716, Perplexity: 1065.9191
Text generated after epoch 5:
The . of the in . the first the the and and . the and . of the first the and . . . Missouri Missouri the the and the and . the Missouri of and the . of the the . of in to the Missouri of the . and

Epoch 6, Loss: 6.9101, Perplexity: 1002.3677
Text generated after epoch 6:
The the in a the and . in the film . to . . a a and of the and of the and and the Missouri and a and of and the first of and and and of . the film . the Missouri and of the first . . the

Epoch 7, Loss: 6.8540, Perplexity: 947.6664
Text generated after epoch 7:
The city to a film to a river and the first . and of the the film in the Missouri of the first and and . . and a Missouri and a the Missouri the of and the the Fleet of the river . the first of the first . the

Epoch 8, Loss: 6.8050, Perplexity: 902.3483
Text generated after epoch 8:
The the Missouri of a Missouri of the Missouri and of a Missouri of the Missouri of the United States of a Missouri of and . and a film . the film . the first States and . . the the Missouri . . . a Missouri and the a and

Epoch 9, Loss: 6.7608, Perplexity: 863.3277
Text generated after epoch 9:
The ship . the film . a first century a episode in the Missouri . the the season . the film to the United . and the first River in the city and the Missouri River in in the and of the first . . the episode . the Missouri .

Epoch 10, Loss: 6.7171, Perplexity: 826.4185
Text generated after epoch 10:
The first and in the Missouri River the first Missouri . the Missouri . . in the first and the Missouri of a Ganges and and . in the in a Missouri and . the Missouri of a . of his Missouri . the . and . the and and .

```

### Generación de Texto

```python
# Generar texto después del entrenamiento
init_text = 'The'  # Texto inicial
generated_text = trainer.generate_text(init_text, length=100, top_k=5)
print("Texto generado después del entrenamiento:")
print(generated_text)
```

```bash
Texto generado después del entrenamiento:
The first of the of . the episode . . . of the shark . . the Missouri of the Missouri in the and . the Missouri States . the the States . in a Missouri of the . of a river in the river and the . in the Missouri . of the Missouri River of the Missouri 's in . in a in . in the Ganges . and the Missouri of the Missouri and the Missouri . and of the . . the Missouri . . . . be a . . the and was the and was
```

## Referencias

- **Embeddings Gaussianos**:

  - [Gaussian Embeddings for Natural Language Processing](https://arxiv.org/abs/1412.6623)

- **PyTorch Documentation**:

  - [Embedding Layer](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
  - [KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)

## Notas

- Asegúrese de ajustar los hiperparámetros según el tamaño y naturaleza de su conjunto de datos.
- El archivo de texto utilizado para el entrenamiento debe estar limpiado y preprocesado adecuadamente.
- El uso de embeddings gaussianos puede aumentar el tiempo de entrenamiento debido al muestreo y cálculo adicional de la pérdida KL.

## Conclusión

Este proyecto demuestra cómo los embeddings gaussianos pueden integrarse con modelos secuenciales como LSTMs para capturar la incertidumbre en secuencias de texto. Al representar palabras como distribuciones en lugar de vectores puntuales, el modelo puede manejar mejor la variabilidad y ambigüedad inherentes al lenguaje natural.
