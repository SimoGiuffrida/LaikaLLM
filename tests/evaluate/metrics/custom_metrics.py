import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def long_tail_coverage(recommended_items, all_interactions, percentile=20):
    """
    Calcola la Coverage della Long-Tail in un sistema di raccomandazione.

    :param recommended_items: Lista di liste con gli item raccomandati per ogni utente
    :param all_interactions: Dizionario {item_id: num_interactions}
    :param percentile: Percentuale di cut-off per definire la long-tail (default 20%)
    :return: Long-Tail Coverage (valore tra 0 e 1)
    """
    interaction_counts = np.array(list(all_interactions.values()))
    cutoff = np.percentile(interaction_counts, percentile)
    long_tail_items = {item for item, count in all_interactions.items() if count <= cutoff}

    recommended_long_tail = set()
    for rec_list in recommended_items:
        recommended_long_tail.update(set(rec_list) & long_tail_items)

    return len(recommended_long_tail) / len(long_tail_items) if long_tail_items else 0.0


def sequence_continuity(recommended_items, item_embeddings):
    """
    Calcola la Continuità della Sequenza tra item raccomandati.

    :param recommended_items: Lista di liste con gli item raccomandati per ogni utente
    :param item_embeddings: Dizionario {item_id: embedding_vector}
    :return: Sequence Continuity Score (valore tra 0 e 1)
    """
    similarities = []

    for rec_list in recommended_items:
        rec_vectors = [item_embeddings[item] for item in rec_list if item in item_embeddings]
        if len(rec_vectors) > 1:
            sim_matrix = cosine_similarity(rec_vectors)
            continuity_score = np.mean([sim_matrix[i, i+1] for i in range(len(rec_vectors)-1)])
            similarities.append(continuity_score)

    return np.mean(similarities) if similarities else 0.0
