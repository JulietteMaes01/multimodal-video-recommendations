#########################################################################
#=============================EVALUATE================================
#########################################################################
def item_based_evaluation(recommendations: Dict[str, List[Dict[str, Any]]], 
                          user_items: Dict[str, Set[str]],
                          user_item_ratings: Dict[str, Dict[str, float]],
                          trailer_to_movie: Dict[str, str],
                          k_values: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Evaluate recommendations using an item-based approach.
    For each movie that users have rated, evaluate the recommendations for that movie.
    """
    print("Performing item-based evaluation...")
    
    # Create an inverse mapping from movie IDs to trailer IDs
    movie_to_trailer = {movie_id: trailer_id for trailer_id, movie_id in trailer_to_movie.items()}
    
    # Initialize metrics
    metrics = {
        'precision': {k: 0.0 for k in k_values},
        'recall': {k: 0.0 for k in k_values},
        'hit_rate': {k: 0.0 for k in k_values},
        'ndcg': {k: 0.0 for k in k_values},
        'MRR': {k: 0.0 for k in k_values}
    }
    
    # Track the number of evaluated items
    evaluated_items = 0
    
    # Create a set of all rated movie IDs
    all_rated_movies = set()
    for user_ratings in user_item_ratings.values():
        all_rated_movies.update(user_ratings.keys())
    
    print(f"Found {len(all_rated_movies)} unique rated movies")
    
    # For each movie that has been rated
    for movie_id in tqdm(all_rated_movies, desc="Evaluating items"):
        # Find the corresponding trailer ID
        trailer_id = movie_to_trailer.get(movie_id)
        
        # Skip if we don't have recommendations for this movie
        if not trailer_id or trailer_id not in recommendations:
            continue
        
        # Get the recommendations for this movie
        movie_recs = recommendations[trailer_id]
        rec_movie_ids = []
        
        # Convert trailer IDs in recommendations to movie IDs
        for rec in movie_recs:
            rec_trailer_id = rec['trailer_id']
            if rec_trailer_id in trailer_to_movie:
                rec_movie_id = trailer_to_movie[rec_trailer_id]
                rec_movie_ids.append(rec_movie_id)
        
        # Find users who have rated this movie positively
        relevant_users = []
        for user_id, relevant_items in user_items.items():
            if movie_id in relevant_items:
                relevant_users.append(user_id)
        
        # Skip if no users rated this movie positively
        if not relevant_users:
            continue
        
        # For each relevant user, evaluate the recommendations
        item_metrics = {
            'precision': {k: 0.0 for k in k_values},
            'recall': {k: 0.0 for k in k_values},
            'hit_rate': {k: 0.0 for k in k_values},
            'ndcg': {k: 0.0 for k in k_values},
            'MRR': {k: 0.0 for k in k_values}

        }
        
        valid_users = 0
        for user_id in relevant_users:
            # Get the set of other movies this user rated positively
            other_relevant_items = user_items[user_id] - {movie_id}
            
            # Skip if user has no other relevant items
            if not other_relevant_items:
                continue
            
            # Get all ratings from this user
            user_ratings = user_item_ratings[user_id]
            
            valid_users += 1
            
            # Calculate metrics for each k
            for k in k_values:
                item_metrics['precision'][k] += precision_at_k(rec_movie_ids, other_relevant_items, k)
                item_metrics['recall'][k] += recall_at_k(rec_movie_ids, other_relevant_items, k)
                item_metrics['hit_rate'][k] += hit_rate_at_k(rec_movie_ids, other_relevant_items, k)
                item_metrics['ndcg'][k] += ndcg_at_k(rec_movie_ids, user_ratings, k)
                item_metrics['MRR'][k] += mean_reciprocal_rank(rec_movie_ids, other_relevant_items, k)
        
        # Average metrics for this item
        if valid_users > 0:
            for metric in item_metrics:
                for k in k_values:
                    item_metrics[metric][k] /= valid_users
            
            # Add to overall metrics
            for metric in metrics:
                for k in k_values:
                    metrics[metric][k] += item_metrics[metric][k]
            
            evaluated_items += 1
    
    # Average metrics across all evaluated items
    if evaluated_items > 0:
        for metric in metrics:
            for k in k_values:
                metrics[metric][k] /= evaluated_items
    
    print(f"Evaluated {evaluated_items} valid items")
    return metrics



#########################################################################
#===============================MAIN======================================
#########################################################################