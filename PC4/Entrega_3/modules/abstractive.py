import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration 

def load_summarizers():
    models = {
        "Pegasus": "google/pegasus-cnn_dailymail",
        "T5": "Overglitch/t5-small-cnn-dailymail",
        "BART": "facebook/bart-large-cnn",
    }
    summarizers = {}
    for model_name, model_path in models.items():
        if model_name == "Pegasus":
            tokenizer = PegasusTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        summarizers[model_name] = (model, tokenizer)
    return summarizers


def abstractive_summary(summarizers, model_name, text, max_length, num_beams):
    model, tokenizer = summarizers[model_name]
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True
    ).to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
