import numpy as np
from w2v_utils import SimilarityCallback
from w2v_utils import initialize_parameters
from w2v_utils import read_data
from w2v_utils import read_glove_vecs
from w2v_utils import relu
from w2v_utils import softmax

words, word_to_vec_map = read_glove_vecs("data/glove.6B.50d.txt")


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the
            formula above.
    """

    if np.all(u == v):
        return 1

    # Compute the dot product between u and v (≈1 line)
    dot = u @ v

    # Compute the L2 norm of u (≈1 line)
    norm_u = np.linalg.norm(u)

    # Compute the L2 norm of v (≈1 line)
    norm_v = np.linalg.norm(v)

    # Avoid division by 0
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0

    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: complete_analogy


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as
    measured by cosine similarity
    """

    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a, e_b, e_c = (
        word_to_vec_map[word_a],
        word_to_vec_map[word_b],
        word_to_vec_map[word_c],
    )

    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None

    for w in words:
        if w == word_c:
            continue

        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word
