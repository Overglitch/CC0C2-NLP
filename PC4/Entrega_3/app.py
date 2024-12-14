import gradio as gr
from modules.extractive import TFIDFSummarizer, TextRankSummarizer, CombinedSummarizer, BERTSummarizer
from modules.abstractive import load_summarizers, abstractive_summary
from modules.preprocessing import Preprocessor, PDFProcessor
from modules.utils import handle_long_text

# Cargar modelos abstractivos finetuneados
summarizers = load_summarizers()

# Función para procesar el archivo cargado
def process_file(file):
    if file is not None:
        pdf_processor = PDFProcessor()
        input_text = pdf_processor.pdf_to_text(file.name)
        return input_text
    return "Por favor, cargue un archivo válido."

# Función principal para generar resúmenes
def summarize(input_text, file, summary_type, method, num_sentences, model_name, max_length, num_beams):
    preprocessor = Preprocessor()

    if file is not None:
        pdf_processor = PDFProcessor()
        input_text = pdf_processor.pdf_to_text(file.name)

    if not input_text:
        return "Por favor, ingrese texto o cargue un archivo válido."

    cleaned_text = preprocessor.clean_text(input_text)

    if summary_type == "Extractivo":
        if method == "TF-IDF":
            summarizer = TFIDFSummarizer()
        elif method == "TextRank":
            summarizer = TextRankSummarizer()
        elif method == "BERT":
            summarizer = BERTSummarizer()
        elif method == "TF-IDF + TextRank":
            summarizer = CombinedSummarizer()
        else:
            return "Método no válido para resumen extractivo."

        return summarizer.summarize(
            preprocessor.split_into_sentences(cleaned_text),
            preprocessor.clean_sentences(preprocessor.split_into_sentences(cleaned_text)),
            num_sentences,
        )

    elif summary_type == "Abstractivo":
        if model_name not in summarizers:
            return "Modelo no disponible para resumen abstractivo."
        return handle_long_text(
            cleaned_text,
            summarizers[model_name][0],
            summarizers[model_name][1],
            max_length=max_length,
            stride=128,
        )

    elif summary_type == "Combinado":
        if model_name not in summarizers:
            return "Modelo no disponible para resumen abstractivo."
        extractive_summary = TFIDFSummarizer().summarize(
            preprocessor.split_into_sentences(cleaned_text),
            preprocessor.clean_sentences(preprocessor.split_into_sentences(cleaned_text)),
            num_sentences,
        )
        return handle_long_text(
            extractive_summary,
            summarizers[model_name][0],
            summarizers[model_name][1],
            max_length=max_length,
            stride=128,
        )

    return "Seleccione un tipo de resumen válido."

# Interfaz dinámica
with gr.Blocks() as interface:
    gr.Markdown("# Aplicación Híbrida para Resumir Documentos de Forma Extractiva y Abstractiva")

    # Entrada de texto o archivo
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(lines=9, label="Ingrese texto", interactive=True)
            # Botón de cargar archivo en la izquierda, debajo de la caja de texto
            load_file_button = gr.Button("Cargar Archivo", visible=False)
        with gr.Column(scale=1):
            file = gr.File(label="Subir archivo (PDF, TXT)")

    # Acción del botón: procesar el archivo y colocar el texto en la caja de texto
    load_file_button.click(
        process_file,
        inputs=[file],
        outputs=[input_text],
    )

    # Mostrar el botón solo cuando se suba un archivo
    def toggle_load_button(file):
        return gr.update(visible=file is not None)

    file.change(
        toggle_load_button,
        inputs=[file],
        outputs=[load_file_button],
    )

    # Selección de tipo de resumen y opciones dinámicas
    summary_type = gr.Radio(
        ["Extractivo", "Abstractivo", "Combinado"],
        label="Tipo de resumen",
        value="Extractivo",
    )
    
    method = gr.Radio(
        ["TF-IDF", "TextRank", "BERT", "TF-IDF + TextRank"],
        label="Método Extractivo",
        visible=True,
    )
    num_sentences = gr.Slider(
        1, 10, value=3, step=1, label="Número de oraciones (Extractivo)", visible=True
    )
    model_name = gr.Radio(
        ["Pegasus", "T5", "BART"],
        label="Modelo Abstractivo",
        visible=False,
    )
    max_length = gr.Slider(
        50, 300, value=128, step=10, label="Longitud máxima (Abstractivo)", visible=False
    )
    num_beams = gr.Slider(
        1, 10, value=4, step=1, label="Número de haces (Abstractivo)", visible=False
    )

    def update_options(summary_type):
        if summary_type == "Extractivo":
            return (
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False))
        elif summary_type == "Abstractivo":
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True))
        elif summary_type == "Combinado":
            return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=True))
        else:
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False))

    summary_type.change(
        update_options,
        inputs=[summary_type],
        outputs=[method, num_sentences, model_name, max_length, num_beams],
    )

    summarize_button = gr.Button("Generar Resumen")
    output = gr.Textbox(lines=10, label="Resumen generado", interactive=True)
    copy_button = gr.Button("Copiar Resumen")

    summarize_button.click(
        summarize,
        inputs=[input_text, file, summary_type, method, num_sentences, model_name, max_length, num_beams],
        outputs=output,
    )

    def copy_summary(summary):
        return summary

    copy_button.click(
        fn=copy_summary,
        inputs=[output],
        outputs=[output],
        js="""function(summary) { navigator.clipboard.writeText(summary); return summary; }""",
    )

if __name__ == "__main__":
    interface.launch()
