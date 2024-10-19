import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from gensim.utils import simple_preprocess
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from nltk.tokenize import word_tokenize
import nltk

generated_title_history = []

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased')
nltk.download('punkt')


def tokenize_and_initialize_bm25(df1):
    tokenized_corpus = [simple_preprocess(doc) for doc in df1['Clean_Content'].dropna()]
    bm25_model = BM25Okapi(tokenized_corpus)
    return bm25_model, tokenized_corpus


# Check if CUDA is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to get BERT embeddings
def get_bert_embedding(text, tokenizer, model, device):
    model.to(device)

    # Truncate to max 512 tokens and apply padding
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # Use the CLS token's embedding for the entire text
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.cpu().detach().numpy()


# BM25 ranking function to get top-n candidates
def bm25_pipeline_faster(bm25_model, query, df1, top_n=10, exclude_titles=None):
    tokenized_query = simple_preprocess(query)
    bm25_scores = bm25_model.get_scores(tokenized_query)

    # Get top candidates based on BM25 scores
    top_indices = np.argsort(bm25_scores)[-top_n * 3:][::-1]
    top_n_candidates_df = df1.iloc[top_indices][['Title', 'Clean_Content', 'Link', 'Initial_Lable', 'category_id']]

    # Exclude already seen titles if provided
    if exclude_titles:
        top_n_candidates_df = top_n_candidates_df[~top_n_candidates_df['Title'].isin(exclude_titles)]

    return top_n_candidates_df.head(top_n)


# BERT-based re-ranking function
def re_rank_with_bert(top_n_contents_df, query, tokenizer, model):
    # Get BERT embedding for the query
    query_embedding = get_bert_embedding(query, tokenizer, model, device)

    # Get embeddings for all documents in the top-n
    document_embeddings = np.array([get_bert_embedding(doc, tokenizer, model, device)
                                    for doc in top_n_contents_df['Clean_Content']])

    # Remove the extra dimension
    document_embeddings = document_embeddings.squeeze(1)

    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

    # Add similarity score to the dataframe and sort by similarity
    top_n_contents_df['similarity'] = similarities
    top_n_contents_df = top_n_contents_df.sort_values(by='similarity', ascending=False)

    return top_n_contents_df


# Hybrid BM25 and BERT ranking pipeline
def bm25_bert_hybrid_pipeline(modified_query, df1, bm25_model, related='no', top_n=10):
    if related == 'no':
        # First query, get top-n BM25 results
        top_10_contents_df = bm25_pipeline_faster(bm25_model, modified_query, df1, top_n=top_n)

        # Track history of generated titles
        if not generated_title_history:
            generated_title_history.extend(top_10_contents_df['Title'].tolist())
        else:
            generated_title_history.clear()
            generated_title_history.extend(top_10_contents_df['Title'].tolist())
    elif related == 'yes':
        # Get related content excluding previously generated titles
        top_10_contents_df = bm25_pipeline_faster(bm25_model, modified_query, df1, top_n=top_n,
                                                  exclude_titles=generated_title_history)
        generated_title_history.extend(top_10_contents_df['Title'].tolist())

    # Re-rank the BM25 results with BERT embeddings
    re_ranked_results = re_rank_with_bert(top_10_contents_df, modified_query, tokenizer, bert_model)

    return re_ranked_results


def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def get_avg_glove_vector(tokens, embeddings_index, vector_size):
    vectors = []
    for token in tokens:
        if token in embeddings_index:
            vectors.append(embeddings_index[token])
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)
