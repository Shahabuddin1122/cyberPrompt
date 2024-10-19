from turtle import pd
from machine_learning.classification import *
from machine_learning.link import get_link
from machine_learning.queries_sequence_maintainance import *
from machine_learning.query_check_and_correct import *
from machine_learning.response import get_response
from machine_learning.summarize import *


def get_final_response(active_query, previous_query, summary):
    current_query = analyze_query(active_query)
    print(f"Active Query: {active_query}")
    print(f"Corrected Query: {current_query}")

    related, modified_query = is_query_related(previous_query, current_query)

    if related:
        related = 'yes'
    else:
        previous_query = ""
        related = 'no'

    df1 = pd.read_csv('data/cleaned_content.csv')
    bm25_model, tokenized_corpus = tokenize_and_initialize_bm25(df1)

    top_10_contents_df = bm25_bert_hybrid_pipeline(modified_query, df1, bm25_model, related=related, top_n=10)

    glove_embeddings = load_glove_embeddings('model/glove.6B.300d.txt')
    top_10_contents_df['Clean_Content_Tokens'] = top_10_contents_df['Clean_Content'].apply(word_tokenize)
    vector_size = 300
    top_10_contents_df['Clean_Content_Vector'] = top_10_contents_df['Clean_Content_Tokens'].apply(
        lambda x: get_avg_glove_vector(x, glove_embeddings, vector_size))

    modified_query_tokens = word_tokenize(modified_query)
    modified_query_vector = get_avg_glove_vector(modified_query_tokens, glove_embeddings, vector_size).reshape(1, -1)

    clean_content_vectors = np.stack(top_10_contents_df['Clean_Content_Vector'].values)
    similarity_scores = cosine_similarity(clean_content_vectors, modified_query_vector).flatten()

    top_10_contents_df['similarity'] = similarity_scores
    top_10_contents_df = top_10_contents_df.sort_values(by='similarity', ascending=False)

    top_match = top_10_contents_df.iloc[0]
    predicted_initial_label = top_match['Initial_Lable']
    predicted_category_id = top_match['category_id']

    relevant_texts = top_10_contents_df['Clean_Content'].tolist()

    batch_size = 4
    summarized_texts = []
    for i in range(0, len(relevant_texts), batch_size):
        batch_content = relevant_texts[i:i + batch_size]
        summaries = summarize_content_with_bart(batch_content, modified_query, max_length=100, min_length=30,
                                                num_beams=6)
        summarized_texts.extend(summaries)

    final_combined_text = ' '.join(summarized_texts)

    input_text = " ".join(summarized_texts)
    prompt = modified_query + " " + input_text
    formatted_response = get_response(prompt)

    link = get_link(summary, top_10_contents_df, summarized_texts)

    # Return the required objects instead of printing
    return related, modified_query, predicted_initial_label, formatted_response, link
