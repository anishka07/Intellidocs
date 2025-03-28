import math
from nltk.tokenize import sent_tokenize, word_tokenize


class TfIdfSummarizer:
    def __init__(self, document: str):
        self.document = document
        self.sentences = sent_tokenize(self.document)
        self.words = [word_tokenize(sentence.lower()) for sentence in self.sentences]
        self.word_set = set(word for sentence in self.words for word in sentence)

    @staticmethod
    def compute_tf(sentence: list[str]) -> dict:
        tf_dict = {}
        word_count = len(sentence)
        for word in sentence:
            if word not in tf_dict:
                tf_dict[word] = 0
            else:
                tf_dict[word] += 1
        tf_dict = {word: count / word_count for word, count in tf_dict.items()}
        return tf_dict

    def compute_idf(self) -> dict:
        idf_dict = {}
        total_sentences = len(self.sentences)
        for word in self.word_set:
            sentence_count_containing_word = sum(
                1 for sentence in self.words if word in sentence
            )
            idf_dict[word] = math.log(
                total_sentences / (1 + sentence_count_containing_word)
            )
        return idf_dict

    def compute_tfidf(self) -> list[dict]:
        idf_dict = self.compute_idf()
        tfidf_scores = []
        for sentence in self.words:
            tf_dict = self.compute_tf(sentence)
            tfidf_dict = {word: tf_dict[word] * idf_dict[word] for word in sentence}
            tfidf_scores.append(tfidf_dict)
        return tfidf_scores

    def rank_sentences(self) -> list[tuple[int, float]]:
        tfidf_scores = self.compute_tfidf()
        sentence_scores = []
        for idx, tfidf_dict in enumerate(tfidf_scores):
            sentence_score = sum(tfidf_dict.values())
            sentence_scores.append((idx, sentence_score))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return sentence_scores

    def summarize(self, num_sentences: int) -> str:
        ranked_sentences = self.rank_sentences()
        selected_sentences_idx = sorted(
            [idx for idx, score in ranked_sentences[:num_sentences]]
        )
        text_summary = " ".join(self.sentences[idx] for idx in selected_sentences_idx)
        return text_summary
