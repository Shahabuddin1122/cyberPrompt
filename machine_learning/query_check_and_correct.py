import language_tool_python
from langdetect import detect, LangDetectException
import logging

tool = language_tool_python.LanguageTool('en-US')


def correct_grammar_spelling(text):
    try:
        matches = tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text
    except Exception as e:
        logging.error(f"Error during grammar correction: {e}")
        return None


def analyze_query(query):
    try:
        lang = detect(query)
    except LangDetectException:
        print("Could not detect language. Please use English words.")
        return None
    except Exception as e:
        logging.error(f"Error during language detection: {e}")
        return None

    logging.info(f"Detected language: {lang}")

    if lang != 'en':
        print(f"Detected language: {lang}. Please use English words or complete the sentence.")
        return None

    corrected_text = correct_grammar_spelling(query)
    return corrected_text
