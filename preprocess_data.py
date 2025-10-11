import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)    
nltk.download('omw-1.4', quiet=True) 
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class Preprocess:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer =WordNetLemmatizer()

    def mergedata(self):
        df_f = pd.read_csv('Data/Fake.csv')
        df_t = pd.read_csv('Data/True.csv')
        df_f['label'] = 0
        df_t['label'] = 1
        df = pd.concat([df_t, df_f], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df['Full_Text'] = (df['title'] + " " + df['text']).str.strip()
        return df

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words] 
        return ' '.join(tokens)
