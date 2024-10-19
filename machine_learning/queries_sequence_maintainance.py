import re

import spacy
from transformers import BertTokenizer, BertForNextSentencePrediction
from difflib import SequenceMatcher
import torch

PRONOUNS = ["it", "this", "that", "these", "those", "he", "she", "they", "his", "her", "their", "its"]

QUESTION_WORDS = {"what", "how", "why", "which", "where", "when", "who", "whose", "whom"}

DESCRIPTIVE_WORDS = {'patching', 'influences', 'malfunctions', 'authentication issue', 'importance', 'role', 'concerns',
                     'scope', 'dangers', 'privacy concern', 'consequences', 'purposes', 'security flaw',
                     'data vulnerability', 'technical difficulty', 'NLP challenge', 'problem', 'issues', 'applications',
                     'availability', 'exploit', 'solutions', 'exploits', 'risk factor', 'buffer overflow', 'mitigation',
                     'attack surface', 'necessities', 'detection rate', 'injection', 'objectives', 'integrity',
                     'drawbacks', 'effects', 'false positive', 'benefits', 'issue', 'significance', 'considerations',
                     'limitation', 'security threat', 'weaknesses', 'danger', 'problems', 'impact', 'opportunities',
                     'backdoor', 'risks', 'implications', 'firewall', 'impacts', 'limitations', 'potential',
                     'exposures', 'confidentiality', 'strengths', 'relevance', 'machine learning issue', 'penetration',
                     'challenges', 'barriers', 'glitches', 'system performance', 'system breach', 'feasibility',
                     'threats', 'risk mitigation', 'vulnerabilities', 'penetration test', 'response', 'data leak',
                     'breaches', 'outcomes', 'bug', 'disadvantages', 'defense', 'security risk', 'factors',
                     'constraints', 'accuracy issue', 'security gap', 'AI challenge', 'advantages', 'AI limitation',
                     'priorities', 'issue mitigation', 'requirements', 'use'}

IRRELEVANT_WORDS = {'tell', 'the', 'example', 'best', 'bit', 'note', 'something', 'thing', 'output', 'the concept',
                    'input', 'type', 'subject', 'detail', 'result', 'process', 'anything', 'another', 'whatever',
                    'other', 'topic', 'section', 'aspect', 'different', 'matter', 'generic', 'context', 'me',
                    'specific', 'particular', 'individual', 'basic', 'general', 'piece', 'information', 'content',
                    'various', 'text', 'element', 'page', 'concept', 'item', 'an example', 'certain', 'more'}

KNOWN_MWES = {
    "cyber attack", "data breach", "network security", "ransomware attack", "phishing scam", "denial of service",
    "distributed denial of service", "social engineering", "firewall configuration",
    "intrusion detection", "identity theft", "vulnerability assessment", "cyber threat intelligence",
    "man-in-the-middle attack", "zero-day vulnerability", "malware infection", "command injection",
    "SQL injection", "cross-site scripting", "buffer overflow", "backdoor attack",
    "brute force attack", "credential stuffing", "password spraying", "honeypot trap",
    "advanced persistent threat", "cyber espionage", "insider threat", "session hijacking",
    "keylogging", "rootkit", "spyware", "adware", "botnet", "cloud misconfiguration",
    "remote code execution", "cryptojacking", "data exfiltration", "exploit kit",
    "sandbox evasion", "cyber warfare", "ethical hacking", "penetration testing", "fuzz testing",
    "network intrusion", "black hat hacking", "phishing detection", "security patch",
    "two-factor authentication", "zero trust architecture", "cryptography", "public key infrastructure",
    "machine learning", "cyber security", 'data loss', 'validation loss', 'data breach softwares', 'malware attacks'
}

nlp = spacy.load("en_core_web_sm")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')


def detect_and_preserve_mwes(query):
    mwes = set()
    for mwe in KNOWN_MWES:
        if mwe in query.lower():
            mwes.add(mwe)
            query = query.lower().replace(mwe, "")
    return mwes, query


def extract_topics_using_dependencies(query):
    mwes, processed_query = detect_and_preserve_mwes(query)
    doc = nlp(processed_query)
    topics = set()

    for token in doc:
        if token.pos_ in (
        "NOUN", "PROPN") and token.text.lower() not in DESCRIPTIVE_WORDS and token.text.lower() not in IRRELEVANT_WORDS:
            topics.add(token.text.strip())

    topics.update(mwes)

    return list(topics)


def expand_current_topics_with_previous(current_topics, previous_topics, current_query):
    for topic in previous_topics:
        if topic.lower() in current_query.lower() and topic.lower() not in current_topics:
            current_topics.append(topic)
    return current_topics


def detect_pronouns(query):
    doc = nlp(query)
    pronouns = [token.text for token in doc if token.pos_ == "PRON" and token.text.lower() in PRONOUNS]
    return pronouns if pronouns else None


def replace_pronouns_in_query(current_query, main_topics_str):
    placeholder_replaced = False
    for placeholder in PRONOUNS:
        if re.search(rf"\b{placeholder}\b", current_query, flags=re.IGNORECASE):
            current_query = re.sub(rf"\b{placeholder}\b", main_topics_str, current_query, flags=re.IGNORECASE)
            placeholder_replaced = True
    return current_query, placeholder_replaced


def calculate_similarity(previous_query, current_query):
    return SequenceMatcher(None, previous_query.lower(), current_query.lower()).ratio()


def calculate_nsp_score(previous_query, current_query):
    inputs = tokenizer.encode_plus(previous_query, current_query, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    is_next_prob = probs[0][0].item()
    return is_next_prob


def is_query_related(previous_query, current_query):
    if previous_query != "":

        previous_topics = extract_topics_using_dependencies(previous_query)
        current_topics = extract_topics_using_dependencies(current_query)

        current_topics = expand_current_topics_with_previous(current_topics, previous_topics, current_query)

        previous_topics_str = ", ".join(previous_topics)
        current_topics_str = ", ".join(current_topics)

        print(f"Previous Topics: {previous_topics_str}")
        print(f"Current Topics: {current_topics_str}")

        #1. Condition: Check if there is a keyword match and a pronoun
        detected_pronouns = detect_pronouns(current_query)
        if previous_topics and current_topics and any(
                topic in current_topics for topic in previous_topics) and detected_pronouns:
            print(f"Pronouns detected: {', '.join(detected_pronouns)}")
            modified_query, _ = replace_pronouns_in_query(current_query, " and ".join(current_topics))
            print("Related: First Condition - Keyword match and pronoun present.")
            return True, modified_query

        #2. Condition: If no keyword match but there is a pronoun, not related
        if previous_topics and current_topics and not any(
                topic in current_topics for topic in previous_topics) and detected_pronouns:
            print(f"Pronouns detected: {', '.join(detected_pronouns)}")
            modified_query, _ = replace_pronouns_in_query(current_query, " and ".join(current_topics))
            print("Not Related: Second Condition - No keyword match but pronoun detected.")
            return False, modified_query

        #3. Condition: If no keyword match, check for pronouns
        if not current_topics and detected_pronouns:
            modified_query, _ = replace_pronouns_in_query(current_query, " and ".join(previous_topics))
            print("Related: Third Condition - No keyword match but pronoun present.")
            return True, modified_query

        #4. Condition: If no pronouns and no valid topics
        if not current_topics:
            modified_query = current_query + " of " + " and ".join(previous_topics)
            print("Related: Fourth Condition - No valid topics and no pronouns.")
            return True, modified_query
        elif current_topics:
            modified_query = current_query
            print("Not Related: Fourth Condition - Valid topics present in current query.")
            return False, modified_query

        #5. Condition: If all other conditions fail, use similarity or NSP score
        similarity_score = calculate_similarity(previous_query, current_query)
        if similarity_score > 0.85:  # Similarity threshold
            modified_query = current_query
            print(f"Related: Fifth Condition - Similarity score indicates relatedness. Score: {similarity_score:.2f}")
            return True, modified_query

        print(f"Not Related: Fifth Condition - Similarity score too low. Score: {similarity_score:.2f}")
        return False, current_query

    else:
        #If previous query is empty and the current query has pronouns and keywords
        detected_pronouns = detect_pronouns(current_query)
        current_topics = extract_topics_using_dependencies(current_query)
        if detected_pronouns and current_topics:
            modified_query, _ = replace_pronouns_in_query(current_query, " and ".join(current_topics))
            print(f"Pronouns detected: {', '.join(detected_pronouns)}")
            previous_query = modified_query
            return False, modified_query
        else:
            modified_query = current_query
            previous_query = current_query
            return False, modified_query
