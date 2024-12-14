import os
import re
import shutil
import time
from pathlib import Path
from datetime import date
from cleantext import clean
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from spellchecker import SpellChecker
import nltk

nltk.data.path.append('/home/user/nltk_data')
nltk.download('punkt')
nltk.download('punkt_tab')


class Preprocessor:
    """Clase para preprocesar texto, realizar limpieza y correcciones."""

    def __init__(self):
        self.spell_checker = SpellChecker()

    @staticmethod
    def clean_text(text: str, lower: bool = False, lang: str = "en") -> str:
        """
        Limpia texto de ruido y caracteres no deseados.
        """
        return clean(
            text,
            fix_unicode=True,
            to_ascii=True,
            lower=lower,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=True,
            no_punct=False,
            lang=lang,
        )

    @staticmethod
    def correct_spacing(text: str, exceptions=None) -> str:
        """
        Corrige espacios alrededor de signos de puntuación y excepciones.
        """
        if exceptions is None:
            exceptions = ["e.g.", "i.e.", "etc.", "cf.", "vs.", "p."]

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r'\s([?.!"](?:\s|$))', r"\1", text)
        text = re.sub(r"\s,", r",", text)

        for exception in exceptions:
            text = text.replace(" ".join(exception.split()), exception)

        return text.strip()

    @staticmethod
    def split_into_sentences(text: str) -> list:
        """
        Divide texto en oraciones usando NLTK.
        """
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    def correct_spelling(self, text: str) -> str:
        """
        Corrige la ortografía del texto dado.
        """
        words = text.split()
        corrected_words = [self.spell_checker.correction(word) for word in words]
        return " ".join(corrected_words)

    def preprocess_text(self, text: str) -> str:
        """
        Limpia, corrige ortografía y ajusta espacios en texto.
        """
        cleaned = self.clean_text(text)
        corrected = self.correct_spelling(cleaned)
        return self.correct_spacing(corrected)
    
    def clean_sentences(self, sentences: list) -> list:
        """
        Limpia cada oración en una lista de oraciones.
        """
        return [self.clean_text(sentence) for sentence in sentences]


class PDFProcessor:
    """Clase para procesar archivos PDF y convertirlos a texto."""

    def __init__(self, max_pages=20):
        self.ocr_model = ocr_predictor(pretrained=True)
        self.max_pages = max_pages

    def pdf_to_text(self, file_path: str) -> str:
        """
        Convierte un archivo PDF a texto usando OCR.
        """
        pdf_file = Path(file_path)
        doc = DocumentFile.from_pdf(pdf_file)
        
        # Asegúrate de que `doc` sea un objeto compatible con pages
        if isinstance(doc, list):
            pages = doc[:self.max_pages] if len(doc) > self.max_pages else doc
        elif hasattr(doc, "pages"):
            pages = doc.pages[:self.max_pages] if len(doc.pages) > self.max_pages else doc.pages
        else:
            raise ValueError("Formato inesperado para el documento PDF.")
        
        raw_text = "\n".join(
            [block.text for page in pages for block in page.blocks]
        )
        return Preprocessor().preprocess_text(raw_text)



class FileHandler:
    """Clase para manejar archivos temporales y limpieza."""

    @staticmethod
    def save_temp_file(file_obj, temp_dir: Path = None) -> str:
        """
        Guarda un archivo temporalmente y retorna su ruta.
        """
        if temp_dir is None:
            temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        file_path = Path(file_obj.name)
        temp_path = temp_dir / file_path.name

        with open(temp_path, "wb") as f:
            f.write(file_obj.read())
        return str(temp_path.resolve())

    @staticmethod
    def clear_temp_files(directory="temp", name_contains="RESULT_"):
        """
        Limpia archivos temporales en el directorio especificado.
        """
        temp_dir = Path(directory)
        if not temp_dir.exists():
            return

        for file in temp_dir.iterdir():
            if file.is_file() and name_contains in file.name:
                file.unlink()

    @staticmethod
    def move_to_completed(from_dir: Path, filename: str, completed_dir="completed"):
        """
        Mueve un archivo procesado a la carpeta 'completed'.
        """
        completed_path = from_dir / completed_dir
        completed_path.mkdir(exist_ok=True)
        shutil.move(from_dir / filename, completed_path / filename)
