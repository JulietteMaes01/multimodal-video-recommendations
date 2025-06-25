#########################################################################
#=========================SIAMESEDATAGENERATOR===========================
#########################################################################

class SiameseDataGenerator:
    def __init__(self, bucket_name: str, ratings_path: str, metadata_path: str,
                 frames_prefix: str, output_s3_path_prefix: str,
                 min_pairs_per_movie: int = 5, random_seed: int = 42): 
        self.bucket_name = bucket_name
        self.ratings_path = ratings_path
        self.metadata_path = metadata_path
        self.frames_prefix = frames_prefix
        self.output_s3_path_prefix = output_s3_path_prefix
        self.min_pairs_per_movie = min_pairs_per_movie
        random.seed(random_seed) # For reproducibility
        np.random.seed(random_seed)

        self.s3_client = boto3.client('s3')
        self.ratings_df = None
        self.metadata_df = None
        self.available_movies = set()
        self.user_ratings = defaultdict(dict)
        self.movie_genres = {}
        self.positive_pairs = []
        self.negative_pairs = []
        self.triplets = [] 

    def load_data(self):
        print("Loading data from S3...")
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.ratings_path.replace(f"s3://{self.bucket_name}/", ""))
        self.ratings_df = pd.read_csv(obj['Body'])
        print(f"Loaded {len(self.ratings_df)} ratings")

        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.metadata_path.replace(f"s3://{self.bucket_name}/", ""))
        self.metadata_df = pd.read_csv(obj['Body'])
        if 'movieId' in self.metadata_df.columns:
            self.metadata_df.set_index('movieId', inplace=True, drop=False) 
        print(f"Loaded metadata for {len(self.metadata_df)} movies")

        self._get_available_movies()
        self._filter_data()
        self._prepare_data_structures()

    def _get_available_movies(self):
        print("Getting available movies from S3 frame tensors...")
        paginator = self.s3_client.get_paginator('list_objects_v2')
        s3_prefix_path = self.frames_prefix.replace(f"s3://{self.bucket_name}/", "")
        
        movie_ids = set()
        print(f"Scanning S3 prefix: {s3_prefix_path}")
        for page in tqdm(paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix_path), desc="S3 pages for frames"):
            if 'Contents' in page:
                for item in page['Contents']:
                    key = item['Key']
                    if key.endswith('_frames.pt'):
                        try:
                            movie_id_str = os.path.basename(key).split('_frames.pt')[0]
                            if movie_id_str.isdigit():
                                movie_ids.add(int(movie_id_str))
                        except:
                            pass # Ignore malformed filenames
        self.available_movies = movie_ids
        print(f"Found {len(self.available_movies)} available movies with frame tensors.")


    def _filter_data(self):
        print("Filtering data based on available movies...")
        initial_ratings = len(self.ratings_df)
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(self.available_movies)].copy()
        print(f"Filtered ratings: {initial_ratings} -> {len(self.ratings_df)}")

        initial_metadata = len(self.metadata_df)
        self.metadata_df = self.metadata_df[self.metadata_df['movieId'].isin(self.available_movies)].copy()
        print(f"Filtered metadata: {initial_metadata} -> {len(self.metadata_df)}")

        movies_with_ratings = set(self.ratings_df['movieId'].unique())
        movies_with_metadata = set(self.metadata_df['movieId'].unique())
        
        # Final available_movies are those present in frames, ratings, and metadata
        self.available_movies = self.available_movies.intersection(movies_with_ratings, movies_with_metadata)
        print(f"Final available movies (in frames, ratings, metadata): {len(self.available_movies)}")
        
        # Further filter DataFrames to ensure consistency
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(self.available_movies)].copy()
        self.metadata_df = self.metadata_df[self.metadata_df['movieId'].isin(self.available_movies)].copy()


    def _prepare_data_structures(self):
        print("Preparing data structures...")
        for _, row in tqdm(self.ratings_df.iterrows(), total=len(self.ratings_df), desc="Processing ratings"):
            user_id = int(row['userId'])
            movie_id = int(row['movieId'])
            if movie_id not in self.available_movies:
                continue
            liked = row['rating'] >= 4.0 # Consistent threshold
            self.user_ratings[user_id][movie_id] = liked

        for movie_id, row in tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df), desc="Processing genres"):
            if movie_id not in self.available_movies: 
                continue
            genres_str = row.get('genres', '')
            self.movie_genres[movie_id] = set(genres_str.split('|')) if pd.notna(genres_str) and genres_str else set()
        print(f"Prepared data for {len(self.user_ratings)} users and {len(self.movie_genres)} movies.")

    def _calculate_genre_jaccard(self, movie1_id: int, movie2_id: int) -> float:
        genres1 = self.movie_genres.get(movie1_id, set())
        genres2 = self.movie_genres.get(movie2_id, set())
        if not genres1 and not genres2:
            return 0.0 
        intersection_size = len(genres1.intersection(genres2))
        union_size = len(genres1.union(genres2))
        if union_size == 0:
            return 0.0
        return intersection_size / union_size

    def generate_positive_pairs(self, min_genre_jaccard_for_positive: float = 0.25):
        print(f"Generating positive pairs (min Jaccard: {min_genre_jaccard_for_positive})...")
        positive_pairs_set = set()
        movie_pair_count = Counter()

        for user_id, ratings in tqdm(self.user_ratings.items(), desc="Users for positive pairs"):
            liked_movies = [mid for mid, liked in ratings.items() if liked and mid in self.movie_genres]
            if len(liked_movies) < 2:
                continue
            for m1, m2 in combinations(liked_movies, 2):
                if self._calculate_genre_jaccard(m1, m2) >= min_genre_jaccard_for_positive:
                    pair = tuple(sorted((m1, m2)))
                    positive_pairs_set.add(pair)
                    movie_pair_count[m1] += 1
                    movie_pair_count[m2] += 1
        
        self.positive_pairs = list(positive_pairs_set)
        print(f"Generated {len(self.positive_pairs)} positive pairs.")
        return movie_pair_count

    def generate_negative_pairs(self, target_count: int = None,
                                max_genre_jaccard_for_negative: float = 0.1,
                                min_user_disagreement_score: int = 1):
        print(f"Generating negative pairs (max Jaccard: {max_genre_jaccard_for_negative}, min disagreement: {min_user_disagreement_score})...")
        if target_count is None:
            target_count = len(self.positive_pairs)
        
        positive_pairs_set = set(self.positive_pairs)
        candidate_negatives_scores = defaultdict(int)

        # Strategy:
        # 1. User-based evidence: Find pairs where users disagreed (liked one, disliked other)
        #    AND these pairs have low genre similarity.
        print("Scoring negatives based on user disagreement & low genre Jaccard...")
        for user_id, ratings in tqdm(self.user_ratings.items(), desc="Users for negative candidates"):
            rated_movie_ids = list(m_id for m_id in ratings.keys() if m_id in self.movie_genres) # Ensure movie has genres
            if len(rated_movie_ids) < 2:
                continue
            for m1, m2 in combinations(rated_movie_ids, 2):
                pair = tuple(sorted((m1, m2)))
                if pair in positive_pairs_set:
                    continue

                genre_jaccard = self._calculate_genre_jaccard(m1, m2)
                if genre_jaccard > max_genre_jaccard_for_negative:
                    continue # Too similar in genre to be a good negative

                m1_liked = ratings[m1]
                m2_liked = ratings[m2]

                if (m1_liked and not m2_liked) or (not m1_liked and m2_liked):
                    candidate_negatives_scores[pair] += 2 # Strong disagreement
                elif not m1_liked and not m2_liked: # Both disliked
                    candidate_negatives_scores[pair] += 1 # Weaker negative evidence

        # Filter by min_user_disagreement_score and prepare for sampling
        scored_potential_negatives = [
            (pair, score) for pair, score in candidate_negatives_scores.items()
            if score >= min_user_disagreement_score
        ]
        print(f"Found {len(scored_potential_negatives)} candidates from user disagreement strategy.")

        # Sort by score (descending) then shuffle to break ties for sampling
        random.shuffle(scored_potential_negatives)
        scored_potential_negatives.sort(key=lambda x: x[1], reverse=True)
        
        self.negative_pairs = [pair for pair, score in scored_potential_negatives]
        
        # Supplement if not enough pairs
        num_needed_supplement = target_count - len(self.negative_pairs)
        if num_needed_supplement > 0:
            print(f"Need to supplement {num_needed_supplement} more negative pairs.")
            all_movie_ids_list = list(self.available_movies)
            current_neg_set = set(self.negative_pairs)
            supplement_attempts = 0
            max_supplement_attempts = num_needed_supplement * 200 # Limit attempts

            pbar_supplement = tqdm(total=num_needed_supplement, desc="Supplementing random negatives")
            while len(self.negative_pairs) < target_count and supplement_attempts < max_supplement_attempts:
                supplement_attempts += 1
                m1, m2 = random.sample(all_movie_ids_list, 2)
                pair = tuple(sorted((m1, m2)))

                if pair not in positive_pairs_set and pair not in current_neg_set:
                    if self._calculate_genre_jaccard(m1, m2) <= max_genre_jaccard_for_negative:
                        self.negative_pairs.append(pair)
                        current_neg_set.add(pair)
                        pbar_supplement.update(1)
            pbar_supplement.close()

        # If still over target, trim. If under, that's the max we could get.
        if len(self.negative_pairs) > target_count:
            self.negative_pairs = random.sample(self.negative_pairs, target_count)
        
        movie_pair_count = Counter()
        for pair in self.negative_pairs:
            movie_pair_count[pair[0]] += 1
            movie_pair_count[pair[1]] += 1
            
        print(f"Generated {len(self.negative_pairs)} negative pairs in total.")
        return movie_pair_count

    def generate_triplets(self, num_neg_per_anchor_positive=1):
        print(f"Generating triplets ({num_neg_per_anchor_positive} negatives per anchor-positive)...")
        self.triplets = []
        
        # Create a quick lookup for movie genres
        # And a list of all movies for random negative sampling
        all_movies_with_genres = list(mid for mid in self.available_movies if mid in self.movie_genres)

        for anchor_id, positive_id in tqdm(self.positive_pairs, desc="Generating triplets"):
            if anchor_id not in self.movie_genres or positive_id not in self.movie_genres:
                continue

            negatives_found_for_this_ap = 0
            # Try to find "harder" negatives: different from anchor but might share *some* characteristic with anchor, but definitely different from positive based on genre.
            # This is a simplified hard negative mining.
            
            shuffled_candidates = random.sample(all_movies_with_genres, len(all_movies_with_genres))

            for negative_candidate_id in shuffled_candidates:
                if negatives_found_for_this_ap >= num_neg_per_anchor_positive:
                    break
                
                if negative_candidate_id == anchor_id or negative_candidate_id == positive_id:
                    continue

                # Condition for a "good" negative in a triplet (A, P, N):
                # 1. N is not P.
                # 2. N is contextually different from P (e.g., low genre similarity with P).
                # 3. (Optional, for harder negatives) N might share some similarity with A, but less than P does.
                # For simplicity here, we'll focus on N being different from P by genre.
                
                # Anchor and Positive are similar (by definition of positive_pairs)
                # Negative should be dissimilar to the Anchor (and implicitly Positive)
                # A stricter condition might be _calculate_genre_jaccard(anchor_id, negative_candidate_id) < threshold
                # AND _calculate_genre_jaccard(positive_id, negative_candidate_id) < threshold

                # Simpler: Ensure negative is quite different from the positive one based on genre
                if self._calculate_genre_jaccard(positive_id, negative_candidate_id) <= 0.1: # N very different from P
                    # And (optionally) ensure N is not *too* similar to A either
                    if self._calculate_genre_jaccard(anchor_id, negative_candidate_id) <= 0.5: # N not overly similar to A
                        self.triplets.append((anchor_id, positive_id, negative_candidate_id))
                        negatives_found_for_this_ap += 1
                        
        print(f"Generated {len(self.triplets)} triplets.")


    def ensure_minimum_pairs_per_movie(self, positive_counts: Counter, negative_counts: Counter):
        print(f"Ensuring each movie appears in at least {self.min_pairs_per_movie} pairs...")
        
        all_movies_in_pairs = set(positive_counts.keys()).union(set(negative_counts.keys()))
        movies_list = list(self.available_movies) # All movies that have frames, ratings, metadata

        for movie_id in tqdm(movies_list, desc="Checking min pairs per movie"):
            current_pos_count = positive_counts[movie_id]
            current_neg_count = negative_counts[movie_id]
            total_current_pairs = current_pos_count + current_neg_count
            
            needed = self.min_pairs_per_movie - total_current_pairs
            if needed <= 0:
                continue

            added_count = 0
            random.shuffle(movies_list) # Shuffle to get varied partners

            for other_movie_id in movies_list:
                if added_count >= needed: break
                if movie_id == other_movie_id: continue
                
                pair = tuple(sorted((movie_id, other_movie_id)))
                is_positive_candidate = self._calculate_genre_jaccard(movie_id, other_movie_id) >= 0.25 # Threshold from generate_positive_pairs
                
                user_co_liked = False
                for user_ratings_dict in self.user_ratings.values():
                    if user_ratings_dict.get(movie_id, False) and user_ratings_dict.get(other_movie_id, False):
                        user_co_liked = True
                        break
                
                if is_positive_candidate and user_co_liked and pair not in self.positive_pairs:
                    self.positive_pairs.append(pair)
                    positive_counts[movie_id] += 1
                    positive_counts[other_movie_id] += 1
                    added_count +=1

            for other_movie_id in movies_list:
                if added_count >= needed: break
                if movie_id == other_movie_id: continue

                pair = tuple(sorted((movie_id, other_movie_id)))
                is_negative_candidate = self._calculate_genre_jaccard(movie_id, other_movie_id) <= 0.1 # Threshold from generate_negative_pairs

                if is_negative_candidate and pair not in self.positive_pairs and pair not in self.negative_pairs:
                    self.negative_pairs.append(pair)
                    negative_counts[movie_id] += 1
                    negative_counts[other_movie_id] += 1
                    added_count += 1
            
        print(f"Final counts after ensuring min pairs - Positive: {len(self.positive_pairs)}, Negative: {len(self.negative_pairs)}")


    def create_train_eval_test_splits(self, train_ratio=0.7, eval_ratio=0.15):
        print("Creating train/eval/test splits...")
        test_ratio = 1.0 - train_ratio - eval_ratio
        assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        all_labeled_pairs = [(pair, 1) for pair in self.positive_pairs] + \
                            [(pair, 0) for pair in self.negative_pairs]
        random.shuffle(all_labeled_pairs)

        total_pairs = len(all_labeled_pairs)
        train_end_idx = int(total_pairs * train_ratio)
        eval_end_idx = train_end_idx + int(total_pairs * eval_ratio)

        train_data = all_labeled_pairs[:train_end_idx]
        eval_data = all_labeled_pairs[train_end_idx:eval_end_idx]
        test_data = all_labeled_pairs[eval_end_idx:]

        print(f"Split sizes - Train: {len(train_data)}, Eval: {len(eval_data)}, Test: {len(test_data)}")
        return {'train': train_data, 'validation': eval_data, 'test': test_data} # Use 'validation' to match PairDataset


    def get_movie_info(self, movie_id: int) -> str:
        try:
            if movie_id in self.metadata_df.index:
                row = self.metadata_df.loc[movie_id]
                title = row.get('title', 'Unknown Title')
                year_val = row.get('year', 'N/A')
                # Ensure year is treated as a string, handling potential float if NaN was present then filled
                year_str = str(int(year_val)) if pd.notna(year_val) and isinstance(year_val, (float, int)) else str(year_val)
                genres = row.get('genres', 'Unknown Genres')
                return f"{title} ({year_str}) - {genres}"
            else:
                return f"Movie ID {movie_id} - Metadata not found"
        except KeyError:
            return f"Movie ID {movie_id} - Metadata lookup error"
        except Exception as e:
            return f"Movie ID {movie_id} - Error getting info: {e}"

    def show_samples(self, num_samples=5):
        print(f"\n=== SAMPLE DATA ({num_samples} samples) ===")
        if self.positive_pairs:
            print(f"\nSample Positive Pairs (movieId1, movieId2):")
            for pair in random.sample(self.positive_pairs, min(num_samples, len(self.positive_pairs))):
                print(f"  ({pair[0]}, {pair[1]})")
                print(f"    M1: {self.get_movie_info(pair[0])}")
                print(f"    M2: {self.get_movie_info(pair[1])}")
        if self.negative_pairs:
            print(f"\nSample Negative Pairs (movieId1, movieId2):")
            for pair in random.sample(self.negative_pairs, min(num_samples, len(self.negative_pairs))):
                print(f"  ({pair[0]}, {pair[1]})")
                print(f"    M1: {self.get_movie_info(pair[0])}")
                print(f"    M2: {self.get_movie_info(pair[1])}")
        if self.triplets:
            print(f"\nSample Triplets (anchor, positive, negative):")
            for triplet in random.sample(self.triplets, min(num_samples, len(self.triplets))):
                print(f"  ({triplet[0]}, {triplet[1]}, {triplet[2]})")
                print(f"    Anchor:   {self.get_movie_info(triplet[0])}")
                print(f"    Positive: {self.get_movie_info(triplet[1])}")
                print(f"    Negative: {self.get_movie_info(triplet[2])}")

    def calculate_statistics(self):
        print(f"\n=== STATISTICS ===")
        movie_pair_counts = Counter()
        all_unique_pairs = set(self.positive_pairs).union(set(self.negative_pairs))
        for pair in all_unique_pairs:
            movie_pair_counts[pair[0]] += 1
            movie_pair_counts[pair[1]] += 1
        
        pair_counts_values = list(movie_pair_counts.values())
        avg_pairs = np.mean(pair_counts_values) if pair_counts_values else 0
        min_pairs = min(pair_counts_values) if pair_counts_values else 0
        max_pairs = max(pair_counts_values) if pair_counts_values else 0

        print(f"Total unique movies in generated pairs: {len(movie_pair_counts)}")
        print(f"Total positive pairs: {len(self.positive_pairs)}")
        print(f"Total negative pairs: {len(self.negative_pairs)}")
        print(f"Total unique pairs generated: {len(all_unique_pairs)}")
        print(f"Total triplets: {len(self.triplets)}")
        print(f"Average pairs per movie involved: {avg_pairs:.2f}")
        print(f"Min pairs for an involved movie: {min_pairs}")
        print(f"Max pairs for an involved movie: {max_pairs}")
        
        movies_with_min_pairs = sum(1 for count in pair_counts_values if count >= self.min_pairs_per_movie)
        print(f"Movies involved in at least {self.min_pairs_per_movie} pairs: {movies_with_min_pairs}/{len(movie_pair_counts)}")

    def _save_splits_to_pkl(self, splits_data_with_labels):
        print("Saving pair splits to S3 as pickle files...")
        # Correctly parse S3 prefix from output_s3_path_prefix
        if self.output_s3_path_prefix.startswith("s3://"):
            path_parts = self.output_s3_path_prefix.replace("s3://", "").split("/", 1)
            bucket = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""
        else: # Assume it's just a prefix if s3:// is missing (for local testing)
            bucket = self.bucket_name 
            prefix = self.output_s3_path_prefix

        if prefix and not prefix.endswith('/'):
            prefix += '/'

        for split_name, pairs_with_labels_list in splits_data_with_labels.items():
            positive_split_pairs = [pair_tuple for pair_tuple, label in pairs_with_labels_list if label == 1]
            negative_split_pairs = [pair_tuple for pair_tuple, label in pairs_with_labels_list if label == 0]

            for pairs_list_to_save, type_suffix in [
                (positive_split_pairs, "positive_pairs"), 
                (negative_split_pairs, "negative_pairs")
            ]:
                if not pairs_list_to_save:
                    print(f"No {type_suffix} to save for {split_name} split.")
                    continue

                s3_key = f"{prefix}{split_name}_{type_suffix}.pkl"
                try:
                    pickle_byte_obj = pickle.dumps(pairs_list_to_save)
                    self.s3_client.put_object(Bucket=bucket, Key=s3_key, Body=pickle_byte_obj)
                    print(f"Saved {len(pairs_list_to_save)} {type_suffix} for {split_name} to s3://{bucket}/{s3_key}")
                except Exception as e:
                    print(f"Error saving {split_name}_{type_suffix}.pkl to S3 for bucket {bucket}, key {s3_key}: {e}")
                    
    def _save_triplets_to_pkl(self, triplets_data_list_of_tuples):
        if not triplets_data_list_of_tuples:
            print("No triplets to save.")
            return
        print("Saving triplets to S3 as pickle file...")
        if self.output_s3_path_prefix.startswith("s3://"):
            path_parts = self.output_s3_path_prefix.replace("s3://", "").split("/", 1)
            bucket = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""
        else:
            bucket = self.bucket_name
            prefix = self.output_s3_path_prefix
            
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        s3_key = f"{prefix}all_triplets.pkl" 
        try:
            pickle_byte_obj = pickle.dumps(triplets_data_list_of_tuples)
            self.s3_client.put_object(Bucket=bucket, Key=s3_key, Body=pickle_byte_obj)
            print(f"Saved {len(triplets_data_list_of_tuples)} triplets to s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"Error saving triplets.pkl to S3 for bucket {bucket}, key {s3_key}: {e}")

    def run_full_pipeline(self):
        print("Starting Siamese Network Data Generation Pipeline")
        print("=" * 60)
        
        self.load_data()
        
        pos_counts = self.generate_positive_pairs(min_genre_jaccard_for_positive=0.25)
        neg_counts = self.generate_negative_pairs(
            target_count=len(self.positive_pairs),
            max_genre_jaccard_for_negative=0.1, # Stricter: Max 10% genre overlap for negatives
            min_user_disagreement_score=1       # Require at least some user disagreement
        )
        
        self.ensure_minimum_pairs_per_movie(pos_counts, neg_counts) 
        
        # Re-balance if ensure_minimum_pairs skewed counts significantly
        if len(self.positive_pairs) != len(self.negative_pairs):
            print("Re-balancing positive and negative pairs after ensuring minimums...")
            if len(self.positive_pairs) > len(self.negative_pairs):
                self.positive_pairs = random.sample(self.positive_pairs, len(self.negative_pairs))
            else:
                self.negative_pairs = random.sample(self.negative_pairs, len(self.positive_pairs))
            print(f"Re-balanced to Positive: {len(self.positive_pairs)}, Negative: {len(self.negative_pairs)}")

        self.generate_triplets(num_neg_per_anchor_positive=1) # Reduced for faster generation initially
        
        splits_data = self.create_train_eval_test_splits()
        
        self.show_samples()
        self.calculate_statistics()
        
        self._save_splits_to_pkl(splits_data)
        self._save_triplets_to_pkl(self.triplets) 

        print("\n" + "=" * 60)
        print("Data generation pipeline completed successfully!")
        return splits_data