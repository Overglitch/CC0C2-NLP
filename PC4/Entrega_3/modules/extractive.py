import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarizer import Summarizer
import networkx as nx


class TFIDFSummarizer:
    @staticmethod
    def summarize(sentences, preprocessed_sentences, num_sentences):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        scores = np.sum(tfidf_matrix.toarray(), axis=1)
        ranked_indices = np.argsort(scores)[::-1]
        return " ".join([sentences[i] for i in ranked_indices[:num_sentences]])


class TextRankSummarizer:
    @staticmethod
    def summarize(sentences, preprocessed_sentences, num_sentences):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        ranked_indices = sorted(scores, key=scores.get, reverse=True)
        return " ".join([sentences[i] for i in ranked_indices[:num_sentences]])


class CombinedSummarizer:
    @staticmethod
    def summarize(sentences, preprocessed_sentences, num_sentences):
        tfidf_summary = TFIDFSummarizer.summarize(
            sentences, preprocessed_sentences, num_sentences
        )
        textrank_summary = TextRankSummarizer.summarize(
            sentences, preprocessed_sentences, num_sentences
        )
        return f"{tfidf_summary} {textrank_summary}"


class BERTSummarizer:
    def __init__(self):
        self.model = Summarizer()

    def summarize(self, text, num_sentences):
        return self.model(text, num_sentences=num_sentences)
