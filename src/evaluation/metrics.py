#########################################################################
#=============================PRECISION@K================================
#########################################################################

def precision_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate precision@k metric."""
    if k == 0 or len(recommended_items) == 0:
        return 0.0
    
    recommended_k = recommended_items[:k]
    num_relevant = sum(1 for item in recommended_k if item in relevant_items)
    
    return num_relevant / min(k, len(recommended_k))


#########################################################################
#==============================RECALL@K==================================
#########################################################################

def recall_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate recall@k metric."""
    if len(relevant_items) == 0:
        return 0.0
    
    recommended_k = recommended_items[:k]
    num_relevant = sum(1 for item in recommended_k if item in relevant_items)
    
    return num_relevant / len(relevant_items)


#########################################################################
#=============================HITRATE@K================================
#########################################################################

def hit_rate_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """
    Calculate hit rate@k metric.
    Hit rate is 1 if at least one relevant item is in the top-k recommendations, 0 otherwise.
    """
    recommended_k = recommended_items[:k]
    for item in recommended_k:
        if item in relevant_items:
            return 1.0
    
    return 0.0

#########################################################################
#=============================NDCG@K================================
#########################################################################

def dcg_at_k(recommended_items: List[str], item_ratings: Dict[str, float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    if k == 0 or len(recommended_items) == 0:
        return 0.0
    
    recommended_k = recommended_items[:k]
    
    dcg = 0
    for i, item in enumerate(recommended_k):
        if item in item_ratings:
            rel = item_ratings[item]
            dcg += (2 ** rel - 1) / np.log2(i + 2)
    
    return dcg

def idcg_at_k(item_ratings: Dict[str, float], k: int) -> float:
    """Calculate Ideal Discounted Cumulative Gain at k."""
    if k == 0 or len(item_ratings) == 0:
        return 0.0
    
    sorted_ratings = sorted(item_ratings.values(), reverse=True)
    relevant_ratings = sorted_ratings[:min(k, len(sorted_ratings))]
    
    idcg = 0
    for i, rel in enumerate(relevant_ratings):
        idcg += (2 ** rel - 1) / np.log2(i + 2)
    
    return idcg

def ndcg_at_k(recommended_items: List[str], item_ratings: Dict[str, float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    idcg = idcg_at_k(item_ratings, k)
    if idcg == 0:
        return 0.0
    
    dcg = dcg_at_k(recommended_items, item_ratings, k)
    return dcg / idcg

#########################################################################
#=============================MRR@K================================
#########################################################################

def mean_reciprocal_rank(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a list of recommendations.
    
    MRR measures where the first relevant item appears in the recommendation list.
    For each query (user/item), the reciprocal rank is the inverse of the position 
    of the first relevant item in the results.
    
    Parameters:
    recommended_items: List of recommended item IDs
    relevant_items: Set of relevant item IDs
    k: Number of recommendations to consider
    
    Returns:
    float: MRR score (0 if no relevant items found)
    """
    if not relevant_items or not recommended_items:
        return 0.0
    
    rec_items = recommended_items[:k]
    
    for i, item in enumerate(rec_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    
    return 0.0
