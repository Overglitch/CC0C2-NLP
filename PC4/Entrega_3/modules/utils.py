from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm.auto import tqdm


def handle_long_text(
    input_text: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    stride: int = 128,
    batch_length: int = 2048,
    min_batch_length: int = 512,
    **generate_kwargs,
) -> str:
    """
    Maneja textos largos dividiéndolos en segmentos y generando resúmenes para cada uno.

    Args:
        input_text (str): Texto completo a resumir.
        model: Modelo de resumen abstractivo.
        tokenizer: Tokenizador asociado al modelo.
        max_length (int): Longitud máxima del resumen generado por segmento.
        stride (int): Cantidad de tokens que se superponen entre segmentos.
        batch_length (int): Longitud máxima de tokens por segmento.
        min_batch_length (int): Longitud mínima permitida por segmento.
        generate_kwargs: Parámetros adicionales para el modelo de generación.

    Returns:
        str: Resumen final concatenado de todos los segmentos.
    """
    # Validar parámetros de longitud
    if batch_length < min_batch_length:
        batch_length = min_batch_length

    # Tokenizar texto completo en segmentos
    encoded_input = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=batch_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    # Obtener IDs y máscaras de atención
    input_ids = encoded_input["input_ids"]
    attention_masks = encoded_input["attention_mask"]

    # Progresión para múltiples segmentos
    summaries = []
    pbar = tqdm(total=len(input_ids), desc="Procesando segmentos")

    for ids, mask in zip(input_ids, attention_masks):
        # Enviar al dispositivo correcto (CPU/GPU)
        ids = ids.unsqueeze(0).to(model.device)
        mask = mask.unsqueeze(0).to(model.device)

        # Generar resumen para el segmento actual
        outputs = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=max_length,
            no_repeat_ngram_size=3,
            num_beams=4,
            early_stopping=True,
            **generate_kwargs,
        )
        # Decodificar resumen generado
        summary = tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        summaries.append(summary)
        pbar.update()

    pbar.close()

    # Concatenar resúmenes y devolver el texto final
    final_summary = " ".join(summaries)
    return final_summary
