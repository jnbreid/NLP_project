import re
import string
import json
import pandas as pd
from tqdm import tqdm
from nlpaug.augmenter.word.synonym import SynonymAug
from nlpaug.augmenter.word.back_translation import BackTranslationAug
from nlpaug.augmenter.word.random import RandomWordAug
from nlpaug.augmenter.word.context_word_embs import ContextualWordEmbsAug
import random
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def clean_text(text):
    """
    Function to clean text data.
    :param text: The input string containing the text to be cleaned.
    :return: Cleaned text.
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    return text

def load_dataset(split={"train", "test", "dev"}, domain={"rest", "laptop"}):
    with open(f'asc/{domain}/{split}.json', 'r') as file:
        dataset = json.load(file)
    return list(dataset.values())

def clean_dataset(dataset):
    for element in tqdm(dataset, desc="Processing Text"):
        element['sentence'] = clean_text(element['sentence'])
        
def remove_duplicates(dataset):
    unique_data = []
    seen = set()
    
    for item in dataset:
        key = (item['sentence'], item['term'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    return unique_data

def check_duplicate(dataset):
    seen = set()
    for item in dataset:
        key = (item['sentence'], item['term'])
        if key in seen:
            print(key)
            return True
        seen.add(key)
    return False

def augment_dataset(dataset, negative_neutral_polarities = {'negative', 'neutral'}, probabilities = [0.5, 0.25, 0.25]):
    aug_syn = SynonymAug()
    aug_trans = BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-zh', to_model_name='Helsinki-NLP/opus-mt-zh-en')
    aug_word = RandomWordAug(action='swap')
    aug_context = ContextualWordEmbsAug(action='insert',model_path='roberta-base')
    
    augmented_data = []
    for item in tqdm(dataset):
        if item['polarity'] in negative_neutral_polarities:
            sentence = item['sentence']
            augmented_sentences = [sentence]
            n=random.choice([1,2])
            random_number = random.random()
            #if random_number <= probabilities[0]:
                #augmented_sentences += aug_word.augment(sentence)
            #elif random_number <= sum(probabilities[:2]):
                #augmented_sentences += aug_syn.augment(sentence, n=n)
            #elif random_number <= sum(probabilities[:3]):
                #augmented_sentences += aug_context.augment(sentence, n = n)
            if random_number <= sum(probabilities):
                try:
                    augmented_sentences += aug_trans.augment(sentence, n=n)
                except Exception as e:
                    print(f"Error in back-translation: {e}")
            for new_sentence in augmented_sentences:
                augmented_data.append({
                    'polarity': item['polarity'],
                    'term': item['term'],
                    'id': item['id'],
                    'sentence': new_sentence
                })
        else:
            augmented_data.append({
                    'polarity': item['polarity'],
                    'term': item['term'],
                    'id': item['id'],
                    'sentence': item['sentence']
                })
    print("Finished Augmentation")
    return augmented_data

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """
    Map POS tag to the first character lemmatize() accepts.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def tokenize_and_lemmatize(text):
    """
    Function to tokenize and lemmatize the input text.
    :param text: The input string containing the text to be processed.
    :return: A list of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    tokenized_text = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokenized_text]
    return lemmatized_text
def preprocess_text(text):
    cleaned_text = clean_text(text)
    preprocessed_text = tokenize_and_lemmatize(cleaned_text)
    
    return preprocessed_text

def preprocess_dataset(dataset):
    for element in tqdm(dataset, desc="Processing Text"):
        element['sentence'] = preprocess_text(element['sentence'])