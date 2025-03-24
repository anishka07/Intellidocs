import torch


def dot_product_of_vectors(vec1, vec2):
    return torch.dot(vec1, vec2)


def cosine_similarity(vec1, vec2):
    vector_dot_product = dot_product_of_vectors(vec1, vec2)
    euclidean_distance_of_vector1 = torch.sqrt(torch.sum(vec1 ** 2))
    euclidean_distance_of_vector2 = torch.sqrt(torch.sum(vec2 ** 2))
    return vector_dot_product / (euclidean_distance_of_vector1 * euclidean_distance_of_vector2)
