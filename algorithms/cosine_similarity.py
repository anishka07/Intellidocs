import numpy as np


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_of_vector_a = np.linalg.norm(a)
    magnitude_of_vector_b = np.linalg.norm(b)
    similarity_score = dot_product / (magnitude_of_vector_a * magnitude_of_vector_b)
    return similarity_score
