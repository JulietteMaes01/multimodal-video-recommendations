if __name__ == "__main__":
    
    # --- Global Constants ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BUCKET_NAME = 'md-data-content-recommendation'
    TEXT_CSV = 's3://md-data-content-recommendation/cleaned_textual_trailers_dataset_with_languages.csv'
    ABLATION_MODE = 'full'
    NUM_FRAMES_FROM_PT, DINO_DIM, VGGISH_DIM = 16, 384, 128
    SAMPLE_RATE, MAX_AUDIO_LENGTH_SEC = 16000, 10
    DINO_PER_FRAME_EMBEDDING_S3_PREFIX = 'movie_trailers_dino_per_frame_embeddings/'
    AUDIO_VGGISH_WAVE_S3_PREFIX = 'movie_trailers_audio_embeddings_vggish/'

    # === THE CONTROL PANEL FOR ALL EXPERIMENTS ===
    experiments_to_run = {
        "E2E_Pair_Cosine": {
            "model_type": "pair",
            "loss_type": "cosine",
            "fusion_network": "Hybrid",
            "checkpoint_s3_prefix": "n2n-model/25-cosine-with-languages-checkpoints/",
            "hyperparameters": {"lr": 5e-5, "batch_size": 16, "epochs": 50},
            "patience": 8, # increase patience
            "test_mode": True,      
            "test_size": 25000 
        },
        "E2E_Pair_Euclidean": {
            "model_type": "pair",
            "loss_type": "euclidean",
            "fusion_network": "Simple",
            "checkpoint_s3_prefix": "n2n-model/25-euclidean-with-languages-checkpoints/",
            "hyperparameters": {"lr":5e-5, "batch_size": 16, "epochs": 50},
            "patience": 8, # increase patience
            "test_mode": True,
            "test_size": 25000
        },
        "E2E_Triplet_Cosine": {
            "model_type": "triplet",
            "loss_type": "cosine",
            "fusion_network": "Simple",
            "checkpoint_s3_prefix": "n2n-model/25-SEMI-HARD-triplet-cosine-checkpoints/",
            "hyperparameters": {"lr": 1e-4, "batch_size": 16, "epochs": 100},
            "patience": 10, # increase patience
            "test_mode": True,
            "test_size": 25000,
            "resume_from": "n2n-model/25-SEMI-HARD-triplet-cosine-checkpoints//epoch_37.pth" 
        },
        "E2E_Triplet_Euclidean": {
            "model_type": "triplet",
            "loss_type": "euclidean",
            "fusion_network": "UltraSimple",
            "checkpoint_s3_prefix": "n2n-model/25-SEMI-HARD-triplet-euclidean-checkpoints/", # normal = 2e-4, 2- = 1e-4, patience = 10
            "hyperparameters": {"lr": 1e-4, "batch_size": 16, "epochs": 100},
            "patience": 10, # increase patience
            "test_mode": True,
            "test_size": 25000
        },
    }

    # --- Pre-build Vocabulary ONCE ---
    master_text_dataset = TextDataset(TEXT_CSV, lang_vocab=None)
    num_languages = master_text_dataset.num_languages
    shared_lang_vocab = master_text_dataset.lang_vocab

    # To store the final results
    all_experiment_results = {}

    # --- MASTER LOOP ---
    for name, config in experiments_to_run.items():
        print(f"\n{'='*30}\nSTARTING EXPERIMENT: {name}\n{'='*30}")
        config['run_name'] = name
        set_seed(42)

        # 1. Create Dataloaders for this experiment
        loaders, positive_pairs_set = create_dataloaders(config, shared_lang_vocab, num_languages)
        
        # 2. Create Model and Criterion
        model, criterion, optimizer, scheduler = create_model_and_criterion(config, DEVICE, num_languages)

        start_epoch = 0
        if config.get("resume_from"):
            try:
                checkpoint = load_checkpoint_from_s3(BUCKET_NAME, config["resume_from"], map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                
                print(f"✅✅✅ Resuming training from epoch {start_epoch}. Model and optimizer state loaded successfully.")
                # Important: Move the model to the correct device *after* loading the state dict
                model.to(DEVICE)
            except Exception as e:
                print(f"❌❌❌ FAILED to load checkpoint: {e}. Starting training from scratch.")
                start_epoch = 0
        
        # Add start_epoch to the config so the training loop knows where to begin
        config['start_epoch'] = start_epoch

        final_writer = None # Initialize writer
        
        # 3. Call the correct, specific training loop
        if config['model_type'] == 'pair':
            print(">>> Dispatching to PAIR training loop...")
            train_pair_model(
                model, loaders, criterion, optimizer, scheduler, DEVICE, config
            )
        elif config['model_type'] == 'triplet':
            print(">>> Dispatching to TRIPLET training loop...")
            train_triplet_model(
                model, loaders['train'], loaders['val'], positive_pairs_set,
                criterion, optimizer, scheduler, DEVICE, config
            )

        # 4. Generate final plots
        if final_writer:
            print("\nFlushing and closing TensorBoard writer...")
            final_writer.flush()
            final_writer.close()
            time.sleep(2) # Give filesystem a moment
            plot_final_learning_curves(final_writer.log_dir, name)
            
        print(f"\n{'='*30}\nFINISHED EXPERIMENT: {name}\n{'='*30}")

    print("\nAll experiments have been completed.")