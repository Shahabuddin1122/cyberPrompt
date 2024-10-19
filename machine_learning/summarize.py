import torch
from transformers import BartForConditionalGeneration, BartTokenizer


#Checking if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


#loding the BART-base model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


#moving the model to the GPU
model.to(device)


def summarize_content_with_bart(content, query, max_length=100, min_length=30, num_beams=4, batch_size=4):
    input_texts = [f"query: {query} document: {content_part}" for content_part in content]

    inputs = tokenizer(input_texts, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(
        device)  #Move inputs to GPU

    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        min_length=min_length,
        temperature=0.8,
        repetition_penalty=2.9,
        num_beams=num_beams,
        length_penalty=2.0,
        early_stopping=True
    )

    summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    return summaries


