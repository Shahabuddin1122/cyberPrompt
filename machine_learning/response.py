from transformers import BartTokenizer, BartForConditionalGeneration
import torch

from machine_learning.puntuation import get_formated_data

load_directory = 'model/BERT_five_main'

model = BartForConditionalGeneration.from_pretrained(load_directory)
tokenizer = BartTokenizer.from_pretrained(load_directory)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(
        inputs['input_ids'],
        max_length=600,
        min_length=400,
        num_beams=6,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        length_penalty=1.5,
        early_stopping=False
    )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    final_response = get_formated_data(response)
    return final_response
