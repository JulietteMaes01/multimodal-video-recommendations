#########################################################################
#=============================CREATE MODEL================================
#########################################################################

def create_model_from_config(config, device, num_languages):
    """Builds the correct model architecture from a configuration dict."""
    
    #1. Select the Fusion Network
    fusion_config = config['fusion_network']
    visual_dim = 256
    audio_dim = 192
    text_dim = 384
    
    if fusion_config == 'Hybrid':
        fusion_net = HybridFusionNetwork(visual_dim, audio_dim, text_dim, embedding_dim=128)
    elif fusion_config == 'Simple':
        fusion_net = SimpleFusionNetwork(visual_dim, audio_dim, text_dim, embedding_dim=128)
    elif fusion_config == 'UltraSimple':
        fusion_net = UltraSimpleFusionNetwork(visual_dim, audio_dim, text_dim, embedding_dim=128)
    else:
        raise ValueError(f"Unknown fusion network type: {fusion_config}")

    #2. Select the Main Siamese Architecture
    model_type = config['model_type']
    
    visual_module = VisualProcessingModule(use_precomputed_embeddings=True, dinov2_embedding_dim=384)
    audio_module = OptimizedAudioProcessingModule(use_vggish=True, vggish_embedding_dim=128)
    text_module = TextProcessingModule(num_languages=num_languages)

    if model_type == 'pair':
        model = OptimizedPairBasedSiameseNetwork(visual_module, audio_module, text_module, fusion_net)
    elif model_type == 'triplet':
        model = OnlineTripletSiameseNetwork(visual_module, audio_module, text_module, fusion_net)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model.to(device)

#########################################################################
#=========================GENERATE EMBEDDINGS==============================
#########################################################################

def generate_e2e_embeddings(pipeline_name, s3_checkpoint_key, output_embeddings_key, model_config):    
    #1. Instantiate Model
    print(f"Building model architecture for {pipeline_name}...")
    
    master_text_dataset = TextDataset(TEXT_CSV, lang_vocab=None)
    num_languages = master_text_dataset.num_languages
    
    model = create_model_from_config(model_config, DEVICE, num_languages)
    
    #2. Load the trained weights
    print(f"Loading weights from s3://{BUCKET_NAME}/{s3_checkpoint_key}")
    checkpoint = load_checkpoint_from_s3(BUCKET_NAME, s3_checkpoint_key, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    #3. Prepare DataLoader
    all_movie_ids = master_text_dataset.text_df.index.astype(str).tolist()
    inference_dataset = InferenceMovieDataset(all_movie_ids, BUCKET_NAME, TEXT_CSV, ABLATION_MODE, master_text_dataset.lang_vocab)
    inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False, num_workers=4)

    #4. Run Inference
    all_embeddings = []
    print(f"Generating embeddings for {pipeline_name}...")
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc=f"Inferring {pipeline_name}"):
            model_args = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'movie_id'}
            
            if model_config['model_type'] == 'pair':
                embeddings = model.forward_single(**model_args)
            else: # triplet
                embeddings = model(**model_args)
                
            all_embeddings.append(embeddings.cpu().numpy())

    final_embeddings = np.vstack(all_embeddings)

    #5. Save the embeddings
    output_data = {'trailer_ids': all_movie_ids, 'features': final_embeddings}
    local_embeddings_file = f"{pipeline_name}_embeddings.pkl"
    print(f"Saving {len(final_embeddings)} embeddings to {local_embeddings_file}")
    with open(local_embeddings_file, "wb") as f:
        pickle.dump(output_data, f)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_embeddings_file, BUCKET_NAME, output_embeddings_key)
    print(f"Uploaded embeddings for {pipeline_name} to S3.")
    
    return local_embeddings_file



#########################################################################
#=============================GENERATE RECOMMENDATIONS================================
#########################################################################

def generate_recommendations_from_embeddings(pipeline_name, local_embeddings_file, output_recs_key, top_k=10):
    print(f"\\n--- GENERATING RECOMMENDATIONS FOR: {pipeline_name.upper()} ---")
    
    with open(local_embeddings_file, "rb") as f:
        data = pickle.load(f)
    embeddings = data['features']
    ids = data['trailer_ids']
    
    print("Calculating similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    
    recommendations = {}
    print(f"Generating Top-{top_k} recommendations...")
    for i in tqdm(range(len(ids)), desc=f"Recommending {pipeline_name}"):
        sim_scores = sim_matrix[i]
        sorted_indices = np.argsort(-sim_scores)[1:top_k+1]
        
        rec_list = [{'trailer_id': ids[j], 'similarity': float(sim_matrix[i, j])} for j in sorted_indices]
        recommendations[ids[i]] = rec_list
        
    local_recs_file = f"{pipeline_name}_recommendations_top_10.pkl"
    print(f"Saving recommendations to {local_recs_file}")
    with open(local_recs_file, "wb") as f:
        pickle.dump(recommendations, f)
        
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_recs_file, BUCKET_NAME, output_recs_key)
    print(f"Uploaded recommendations to s3://{BUCKET_NAME}/{output_recs_key}")