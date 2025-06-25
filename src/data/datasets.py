#########################################################################
#============================PAIR DATASET================================
#########################################################################

class PairDataset(Dataset):
    """Dataset for pair-based Siamese network"""
    def __init__(self, data_path, bucket_name, text_csv_path, ablation_mode, split='train', lang_vocab=None):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.split = split
        self.ablation_mode = ablation_mode # Store it
        
        # Parse S3 path
        s3_path_parts = data_path.split('/')
        base_bucket = s3_path_parts[2]
        base_prefix = '/'.join(s3_path_parts[3:])

        base_prefix = '/'.join(s3_path_parts[3:])
        base_prefix = base_prefix.rstrip('/')
        
        pos_key = f"{base_prefix}/{split}_positive_pairs.pkl"
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                self.s3_client.download_file(base_bucket, pos_key, temp_file.name)
                with open(temp_file.name, 'rb') as f: 
                    self.positive_pairs = pickle.load(f)
            except Exception as e:
                print(f"Error downloading/loading positive pairs: {e}")
                print(f"Attempted to access: s3://{base_bucket}/{pos_key}")
                raise
        
        # Download negative pairs
        neg_key = f"{base_prefix}/{split}_negative_pairs.pkl"  
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                self.s3_client.download_file(base_bucket, neg_key, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    self.negative_pairs = pickle.load(f)
            except Exception as e:
                print(f"Error downloading/loading negative pairs: {e}")
                print(f"Attempted to access: s3://{base_bucket}/{neg_key}")
                raise
        
        # Combine positive and negative pairs
        self.all_pairs = []
        for pair in self.positive_pairs:
            movie1_id, movie2_id = pair
            self.all_pairs.append((movie1_id, movie2_id, 1))
        for pair in self.negative_pairs:
            movie1_id, movie2_id = pair
            self.all_pairs.append((movie1_id, movie2_id, 0))
            
        # Initialize video and text helpers
        self.video_dataset = VideoDatasetS3(bucket_name)
        if self.ablation_mode == 'full':
            self.text_dataset = TextDataset(text_csv_path, lang_vocab=lang_vocab)
        else:
            self.text_dataset = None
    
    def __len__(self):
        return len(self.all_pairs)

    @lru_cache(maxsize=128) #try32 if not working
    def _cached_get_video_frames(self, video_path):
        return extract_video_frames(video_path)
    
    @lru_cache(maxsize=128)
    def _cached_get_audio_features(self, video_path):
        return extract_audio_features(video_path)
    
    def __getitem__(self, idx):
        movie1_id, movie2_id, label = self.all_pairs[idx]
        
        # --- Visual ---
        # Load the original (normalized) frames for low-level feature calculation
        frames1_for_low_level = self.video_dataset.get_preextracted_frames(movie1_id, num_frames=NUM_FRAMES_FROM_PT)
        frames2_for_low_level = self.video_dataset.get_preextracted_frames(movie2_id, num_frames=NUM_FRAMES_FROM_PT)
        
        # Load pre-extracted DINOv2 per-frame embeddings
        dino_frame_embeddings1 = self.video_dataset.get_preextracted_dino_frame_embeddings(movie1_id)
        dino_frame_embeddings2 = self.video_dataset.get_preextracted_dino_frame_embeddings(movie2_id)

        if torch.isnan(dino_frame_embeddings1).any():
            print(f"!!! NaN found in DINO embedding for movie_id: {movie1_id}")
    
        # --- Audio ---
        vggish_emb1, waveform1 = torch.zeros(VGGISH_DIM), torch.zeros(1, SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC)
        vggish_emb2, waveform2 = torch.zeros(VGGISH_DIM), torch.zeros(1, SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC)
    
        if self.ablation_mode != 'visual_only':
            vggish_emb1, waveform1 = self.video_dataset.get_preextracted_audio_vggish_wave(movie1_id)
            vggish_emb2, waveform2 = self.video_dataset.get_preextracted_audio_vggish_wave(movie2_id)
        
        # --- Text ---
        item = {
            'movie1_id': movie1_id,
            'movie2_id': movie2_id,
            'frames1_for_low_level': frames1_for_low_level, 
            'dino_frame_embeddings1': dino_frame_embeddings1,
            'frames2_for_low_level': frames2_for_low_level,
            'dino_frame_embeddings2': dino_frame_embeddings2,
            'vggish_embedding1': vggish_emb1,                
            'waveform1': waveform1,                          
            'vggish_embedding2': vggish_emb2,
            'waveform2': waveform2,
            'label': torch.tensor(label, dtype=torch.float32)
        }
        
        if self.ablation_mode == 'full' and self.text_dataset is not None:
            text1_data = self.text_dataset.get_text_features(movie1_id)
            item['video1_title'] = text1_data['title']
            item['video1_plot'] = text1_data['plot']
            item['video1_tfidf'] = text1_data['tfidf_features']
            item['video1_year'] = text1_data['year']            
            item['video1_language_idx'] = text1_data['language_idx'] 
            
            text2_data = self.text_dataset.get_text_features(movie2_id)
            item['video2_title'] = text2_data['title']
            item['video2_plot'] = text2_data['plot']
            item['video2_tfidf'] = text2_data['tfidf_features']
            item['video2_year'] = text2_data['year']        
            item['video2_language_idx'] = text2_data['language_idx']

        return item


#########################################################################
#============================TRIPLET DATASET================================
#########################################################################

# =================================================================
# CLASS 1: FOR ONLINE TRAINING (provides Anchor-Positive pairs)
# =================================================================
class TripletDatasetForOnlineTraining(Dataset):
    """
    This dataset is used ONLY for TRAINING.
    It loads POSITIVE pairs and returns a dictionary for the anchor and a dictionary for the positive.
    The negative is found "online" during the training loop.
    """
    def __init__(self, data_path, bucket_name, text_csv_path, ablation_mode, split='train', lang_vocab=None):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.split = split
        self.ablation_mode = ablation_mode

        s3_path_parts = data_path.split('/')
        if len(s3_path_parts) < 3:
             raise ValueError(f"Invalid S3 path format: {data_path}")
        
        base_bucket = s3_path_parts[2]
        base_prefix = '/'.join(s3_path_parts[3:]).rstrip('/')
        
        # This dataset loads POSITIVE pairs to serve as (Anchor, Positive)
        pos_key = f"{base_prefix}/{split}_positive_pairs.pkl"
        
        print(f"Loading (Anchor, Positive) pairs from: s3://{base_bucket}/{pos_key}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            try:
                self.s3_client.download_file(base_bucket, pos_key, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    self.anchor_positive_pairs = pickle.load(f)
                print(f"Successfully loaded {len(self.anchor_positive_pairs)} anchor-positive pairs for '{split}' split.")
            except Exception as e:
                print(f"FATAL ERROR: Could not download/load positive pairs for online training from s3://{base_bucket}/{pos_key}")
                print(f"Error details: {e}")
                raise
            finally:
                os.unlink(temp_file.name)
        
        # Initialize helper classes
        self.video_dataset = VideoDatasetS3(bucket_name)
        if self.ablation_mode == 'full':
            self.text_dataset = TextDataset(text_csv_path, lang_vocab=lang_vocab)
        else:
            self.text_dataset = None

    def __len__(self):
        return len(self.anchor_positive_pairs)
    
    def _get_single_item_features(self, movie_id):
        """Helper function to load all features for a single movie_id."""
        
        # --- Visual Features ---
        frames_for_low_level = self.video_dataset.get_preextracted_frames(movie_id, num_frames=NUM_FRAMES_FROM_PT)
        dino_frame_embeddings = self.video_dataset.get_preextracted_dino_frame_embeddings(movie_id)

        # --- Audio Features ---
        vggish_emb, waveform = torch.zeros(VGGISH_DIM), torch.zeros(1, SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC)
        if self.ablation_mode != 'visual_only':
            vggish_emb, waveform = self.video_dataset.get_preextracted_audio_vggish_wave(movie_id)
        
        # --- Prepare the data dictionary ---
        item_data = {
            'movie_id': torch.tensor(movie_id, dtype=torch.long), # CRITICAL for online mining
            'frames_for_low_level': frames_for_low_level,
            'dino_frame_embeddings': dino_frame_embeddings,
            'vggish_embedding': vggish_emb,
            'waveform_for_spec_cnn': waveform
        }

        # --- Text Features (if applicable) ---
        if self.ablation_mode == 'full' and self.text_dataset is not None:
            text_data = self.text_dataset.get_text_features(movie_id)
            item_data.update(text_data)
        else:
            # Provide None placeholders for model's forward signature if text isn't used
            item_data.update({'title': None, 'plot': None, 'tfidf_features': None, 'year': None, 'language_idx': None})
            
        return item_data

    def __getitem__(self, idx):
        # This returns the data for the anchor and the positive
        anchor_id, positive_id = self.anchor_positive_pairs[idx]
        anchor_data = self._get_single_item_features(anchor_id)
        positive_data = self._get_single_item_features(positive_id)
        
        return {"anchor_data": anchor_data, "positive_data": positive_data}


# ======================================================================
# CLASS 2: FOR VALIDATION (provides pre-generated A, P, N triplets)
# ======================================================================
class TripletDatasetForValidation(Dataset):
    """
    This dataset is used ONLY for VALIDATION.
    It loads the pre-generated, "easy" triplets file to provide a consistent
    benchmark for measuring validation loss.
    """
    def __init__(self, data_path, bucket_name, text_csv_path, ablation_mode, split='validation', lang_vocab=None):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.split = split
        self.ablation_mode = ablation_mode

        s3_path_parts = data_path.split('/')
        if len(s3_path_parts) < 3:
             raise ValueError(f"Invalid S3 path format: {data_path}")

        base_bucket = s3_path_parts[2]
        base_prefix = '/'.join(s3_path_parts[3:]).rstrip('/')
        
        # This dataset loads the pre-generated TRIPLETS
        triplet_key = f"{base_prefix}/{split}_triplets.pkl"
        
        print(f"Loading (A, P, N) triplets from: s3://{base_bucket}/{triplet_key}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            try:
                self.s3_client.download_file(base_bucket, triplet_key, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    self.triplets = pickle.load(f)
                print(f"Successfully loaded {len(self.triplets)} triplets for '{split}' split.")
            except Exception as e:
                print(f"Could not load '{triplet_key}'. Trying 'all_triplets.pkl' instead...")
                triplet_key = f"{base_prefix}/all_triplets.pkl"
                try:
                    self.s3_client.download_file(base_bucket, triplet_key, temp_file.name)
                    with open(temp_file.name, 'rb') as f:
                        all_triplets = pickle.load(f)
                    
                    random.seed(42) # for reproducible splits
                    random.shuffle(all_triplets)
                    train_end = int(0.7 * len(all_triplets))
                    val_end = int(0.85 * len(all_triplets))
                    
                    if split == 'train': self.triplets = all_triplets[:train_end]
                    elif split == 'validation': self.triplets = all_triplets[train_end:val_end]
                    else: self.triplets = all_triplets[val_end:]
                    
                    print(f"Loaded from 'all_triplets.pkl' and took {len(self.triplets)} for '{split}' split.")
                    
                except Exception as e2:
                    print(f"FATAL ERROR: Could not download/load triplets from either key. Error: {e2}")
                    raise
            finally:
                os.unlink(temp_file.name)

        # Initialize helper classes (same as the training dataset)
        self.video_dataset = VideoDatasetS3(bucket_name)
        if self.ablation_mode == 'full':
            self.text_dataset = TextDataset(text_csv_path, lang_vocab=lang_vocab)
        else:
            self.text_dataset = None
    
    def __len__(self):
        return len(self.triplets)

    def _get_single_item_features(self, movie_id):
        """Helper function to load all features for a single movie_id."""
        item_data = {
            'movie_id': torch.tensor(movie_id, dtype=torch.long),
            'frames_for_low_level': self.video_dataset.get_preextracted_frames(movie_id, num_frames=NUM_FRAMES_FROM_PT),
            'dino_frame_embeddings': self.video_dataset.get_preextracted_dino_frame_embeddings(movie_id),
            'vggish_embedding': torch.zeros(VGGISH_DIM),
            'waveform_for_spec_cnn': torch.zeros(1, SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC)
        }
        if self.ablation_mode != 'visual_only':
            item_data['vggish_embedding'], item_data['waveform_for_spec_cnn'] = self.video_dataset.get_preextracted_audio_vggish_wave(movie_id)
        
        if self.ablation_mode == 'full' and self.text_dataset is not None:
            text_data = self.text_dataset.get_text_features(movie_id)
            item_data.update(text_data)
        else:
            item_data.update({'title': None, 'plot': None, 'tfidf_features': None, 'year': None, 'language_idx': None})
            
        return item_data

    def __getitem__(self, idx):
        # This returns the data for the pre-generated anchor, positive, and negative
        anchor_id, positive_id, negative_id = self.triplets[idx]
        anchor_data = self._get_single_item_features(anchor_id)
        positive_data = self._get_single_item_features(positive_id)
        negative_data = self._get_single_item_features(negative_id)
        
        return {
            "anchor_data": anchor_data,
            "positive_data": positive_data,
            "negative_data": negative_data
        }


########################ALLMOVIES###################################

class AllMoviesDataset(Dataset):
    """
    A dataset that returns individual movie items and their movie_id.
    This is used for proper in-batch negative mining.
    """
    def __init__(self, data_path, bucket_name, text_csv_path, ablation_mode, split='train', lang_vocab=None):
        self.ablation_mode = ablation_mode
        
        print(f"Initializing AllMoviesDataset for '{split}' split...")
        s3_client = boto3.client('s3')
        s3_path_parts = data_path.split('/')
        base_bucket = s3_path_parts[2]
        base_prefix = '/'.join(s3_path_parts[3:]).rstrip('/')

        all_movie_ids = set()
        for pair_file in [f"{split}_positive_pairs.pkl", f"{split}_negative_pairs.pkl"]:
            key = f"{base_prefix}/{pair_file}"
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_f:
                    s3_client.download_file(base_bucket, key, temp_f.name)
                    with open(temp_f.name, 'rb') as f:
                        pairs = pickle.load(f)
                    for id1, id2 in pairs:
                        all_movie_ids.add(id1)
                        all_movie_ids.add(id2)
                os.unlink(temp_f.name)
            except Exception as e:
                print(f"Warning: Could not load {key}. It might not exist for this split. Error: {e}")

        self.movie_ids = sorted(list(all_movie_ids))
        print(f"Found {len(self.movie_ids)} unique movies for the '{split}' split.")

        self.video_dataset = VideoDatasetS3(bucket_name)
        if self.ablation_mode == 'full':
            self.text_dataset = TextDataset(text_csv_path, lang_vocab=lang_vocab)
        else:
            self.text_dataset = None

        self._get_single_item_features = TripletDatasetForValidation._get_single_item_features.__get__(self)

    def __len__(self):
        return len(self.movie_ids)

    def __getitem__(self, idx):
        movie_id = self.movie_ids[idx]
        item_data = self._get_single_item_features(movie_id)
        return item_data