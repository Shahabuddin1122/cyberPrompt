from deepmultilingualpunctuation import PunctuationModel
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

punctuation_model = PunctuationModel()


def get_punctuated_data(bart_response):
    sentences = sent_tokenize(bart_response)

    punctuated_sentences = [punctuation_model.restore_punctuation(sentence) for sentence in sentences]

    punctuated_response = " ".join(punctuated_sentences)
    print(f"Punctuated Response: {punctuated_response}")
    return punctuated_response


def capitalize_sentences(text):
    sentences = sent_tokenize(text)
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    return " ".join(capitalized_sentences)


def get_formated_data(text):
    punctuated_response  = get_punctuated_data(text)
    final_response = capitalize_sentences(punctuated_response)
    return final_response
