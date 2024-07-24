import pandas as pd
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import os
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from nltk.corpus import stopwords as nltk_stopwords
from umap import UMAP
"""!pip install langdetect
!pip install umap-learn"""




class Preprocess:
    def __init__(self):
        self.train_df = None
        self.test_df = None

    def read_csv(self,  test_file=None):
        """Read CSV files. If test_file is None, only read train_file."""
        self.train_df = pd.read_csv(os.getcwd() + "/data/train.csv", dtype="str")
        self.test_df = pd.read_csv(os.getcwd() + "/data/test.csv", dtype="str")
        if test_file:
            self.test_df = pd.read_csv(test_file, dtype="str")
        return self.train_df, self.test_df

    def remove_rows(self):
        """Remove rows with missing values and convert to string"""
        self.train_df = self.train_df.dropna()
        self.test_df = self.test_df.dropna()
        self.train_df = self.train_df.astype(str)
        self.test_df = self.test_df.astype(str)
        return self.train_df, self.test_df

    def remove_duplicates(self):
        """Remove duplicate rows"""
        self.train_df = self.train_df.drop_duplicates()
        self.test_df = self.test_df.drop_duplicates()
        return self.train_df, self.test_df

    def remove_rows_lower_than20(self):
        """Remove rows where 'text' length is less than 20"""
        self.train_df = self.train_df[self.train_df["text"].str.len() >= 20]
        self.test_df = self.test_df[self.test_df["text"].str.len() >= 20]
        return self.train_df, self.test_df


    def newtext(
        self,
    ):
        """Convert non-string objects in strings for title, author and text and then fill missing values with empty strings
        Create new column called "new text" merging title, text and author column"""
#        self.train_df["title"] = self.train_df["title"].astype(str)
#        self.train_df["author"] = self.train_df["author"].astype(str)
#        self.train_df["text"] = self.train_df["text"].astype(str)

#        self.train_df["author"] = self.train_df["author"].fillna("")
#        self.train_df["title"] = self.train_df["title"].fillna("")
#        self.train_df["text"] = self.train_df["text"].fillna("")

#        self.test_df["title"] = self.test_df["title"].astype(str)
#        self.test_df["author"] = self.test_df["author"].astype(str)
#        self.test_df["text"] = self.test_df["text"].astype(str)

#        self.test_df["author"] = self.test_df["author"].fillna("")
#        self.test_df["title"] = self.test_df["title"].fillna("")
#        self.test_df["text"] = self.test_df["text"].fillna("")


        self.train_df["new_text"] = self.train_df["text"] + self.train_df["title"] + self.train_df["author"]
        self.test_df["new_text"] = self.test_df["text"] + self.test_df["title"] + self.test_df["author"]

        return self.train_df, self.test_df

    def filter_english_text_edit_df(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Detects and filters only English text from a DataFrame based on a specified text column.
        Edits the original DataFrame to keep only the rows where the detected language is English.

        Args:
        - df (pd.DataFrame): The original DataFrame containing text data.
        - text_column (str): The name of the column in df containing text data to analyze.

        Returns:
        - pd.DataFrame: Filtered DataFrame containing only rows with English text.
        """
        # Validate that the text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        # Initialize an empty list to store indices of rows to keep
        keep_indices = []

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            text = row[text_column]
            try:
                # Detect the language of the text
                if detect(text) == "en":
                    # If the language is English, add the index to the list of keep_indices
                    keep_indices.append(index)
            except LangDetectException as e:
                print(f"Language detection failed for row {index}: {text}")
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error for row {index}: {text}")
                print(f"Error: {e}")

        # Filter the original DataFrame to keep only the rows where text is in English
        filtered_df = df.loc[keep_indices].reset_index(drop=True)

        # Check the number of rows before and after filtering
        print(f"Original DataFrame size: {len(df)}")
        print(f"Filtered DataFrame size: {len(filtered_df)}")

        return filtered_df


class Vectorization:
    def __init__(self) -> None:
        pass
    

    def get_tfidf_vectors(
        self,
        corpus: np.ndarray,
        stop_words: str = None,
        max_features: int = None,
        n: int = 1,
    ) -> np.ndarray:
        """
        Vectorizes a corpus of text using TF-IDF (Term Frequency-Inverse Document Frequency).

        Args:
        - corpus (np.ndarray): Array-like, each element is a string representing a document.
        - stop_words (str, optional): Language for stop words ('english', 'spanish', etc.) or None to include all words.
        - max_features (int, optional): Maximum number of features (terms) to consider when vectorizing.
        - n (int, optional): Range of n-grams to consider; (n, n) means only n-grams of size 'n'.

        Returns:
        - np.ndarray: Matrix of TF-IDF vectors where each row corresponds to a document in the corpus.

        Note:
        This function uses sklearn's TfidfVectorizer to compute TF-IDF vectors.
        Each document in the corpus is transformed into a vector representation based on the TF-IDF scores of its terms.
        """

        # Create a TfidfVectorizer object with the given parameters
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words, max_features=max_features, ngram_range=(n, n)
        )

        # Fit the vectorizer to the corpus and transform the text data into TF-IDF vectors
        self.vectorized = self.vectorizer.fit_transform(corpus)

        # Return the resulting TF-IDF vectors
        return self.vectorized
