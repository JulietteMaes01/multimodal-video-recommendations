#########################################################################
#==============================PAIR MODEL================================
#########################################################################

class OptimizedPairBasedSiameseNetwork(nn.Module):
    def __init__(self, visual_module, audio_module, text_module, fusion_network):
        super().__init__()
        self.visual_module = visual_module
        self.audio_module = audio_module
        self.text_module = text_module
        self.fusion_network = fusion_network
    
    def forward_single(self, 
                       frames_for_low_level,         # For visual low-level
                       dino_frame_embeddings,        # For visual high-level (precomputed)
                       vggish_embedding=None,        # For audio (precomputed VGGish)
                       waveform_for_spec_cnn=None,   # For audio (raw wave for spec CNN)
                       title=None, plot=None, tfidf_features=None, # For text
                       year=None, language_idx=None):

        # 1. Visual Processing
        visual_features = self.visual_module(
            frames=frames_for_low_level,
            precomputed_dino_embeddings=dino_frame_embeddings
        )
        
        # 2. Audio Processing
        audio_features = None
        if self.audio_module is not None:
            audio_features = self.audio_module(
                precomputed_vggish_embedding=vggish_embedding, 
                waveform_for_spec_cnn=waveform_for_spec_cnn
            )
        
        # 3. Text Processing
        text_features = None
        if self.text_module is not None:
            # check that title and plot are not None before passing them
            if title is not None and plot is not None:
                text_features = self.text_module(
                    title=title, 
                    plot=plot, 
                    tfidf_features=tfidf_features, 
                    year=year, 
                    language_idx=language_idx
                )
            else:
                # If title/plot are missing, can't create text features -> create a zero tensor of the correct size to avoid errors.
                print("⚠️ Warning: title or plot missing for text module. Creating zero tensor.")
                batch_size = visual_features.shape[0]
                device = visual_features.device
                text_features = torch.zeros(batch_size, self.text_module.output_dim, device=device)

            
        fused_embedding = self.fusion_network(visual_features, audio_features, text_features)
        return fused_embedding
        
    def forward(self, 
                frames1_for_low_level, dino_frame_embeddings1, waveform1, vggish_embedding1, 
                frames2_for_low_level, dino_frame_embeddings2, waveform2, vggish_embedding2,
                video1_title=None, video1_plot=None, video1_tfidf=None, video1_year=None, video1_language_idx=None,
                video2_title=None, video2_plot=None, video2_tfidf=None, video2_year=None, video2_language_idx=None):
       
        current_batch_size = frames1_for_low_level.shape[0]
        current_device = frames1_for_low_level.device

        nan_check_visual = (
            torch.isnan(frames1_for_low_level).any() or torch.isnan(dino_frame_embeddings1).any() or
            torch.isnan(frames2_for_low_level).any() or torch.isnan(dino_frame_embeddings2).any()
        )
        if nan_check_visual:
            print("⚠️ NaN detected in primary visual input tensors in SNN forward.")
            emb_dim = self.fusion_network.embedding_dim if hasattr(self.fusion_network, 'embedding_dim') else 128
            return (torch.zeros(current_batch_size, emb_dim, device=current_device),
                    torch.zeros(current_batch_size, emb_dim, device=current_device))

        if self.audio_module:
            waveform1 = torch.nan_to_num(waveform1, nan=0.0, posinf=0.0, neginf=0.0) if waveform1 is not None else None
            vggish_embedding1 = torch.nan_to_num(vggish_embedding1, nan=0.0, posinf=0.0, neginf=0.0) if vggish_embedding1 is not None else None
            waveform2 = torch.nan_to_num(waveform2, nan=0.0, posinf=0.0, neginf=0.0) if waveform2 is not None else None
            vggish_embedding2 = torch.nan_to_num(vggish_embedding2, nan=0.0, posinf=0.0, neginf=0.0) if vggish_embedding2 is not None else None
        if self.text_module:
            video1_tfidf = torch.nan_to_num(video1_tfidf, nan=0.0) if video1_tfidf is not None else None
            video2_tfidf = torch.nan_to_num(video2_tfidf, nan=0.0) if video2_tfidf is not None else None


        embedding1 = self.forward_single( 
            frames1_for_low_level, dino_frame_embeddings1,
            vggish_embedding1, waveform1, 
            video1_title, video1_plot, tfidf_features=video1_tfidf,
            year=video1_year, language_idx=video1_language_idx
        )
        
        embedding2 = self.forward_single(
            frames2_for_low_level, dino_frame_embeddings2,
            vggish_embedding2, waveform2, 
            video2_title, video2_plot, tfidf_features=video2_tfidf,
            year=video2_year, language_idx=video2_language_idx 
        )

        return embedding1, embedding2


#########################################################################
#=============================TRIPLET MODEL================================
#########################################################################

class OnlineTripletSiameseNetwork(nn.Module):
    def __init__(self, visual_module, audio_module, text_module, fusion_network):
        """
        Initializes the single processing tower of the Siamese network.
        Its only job is to process ONE batch of movies into embeddings.
        """
        super().__init__()
        self.visual_module = visual_module
        self.audio_module = audio_module
        self.text_module = text_module
        self.fusion_network = fusion_network

    def forward(self, **batch_data):
        """
        Processes a BATCH of movie data into a BATCH of embeddings.

        Args:
            **batch_data: A dictionary of features. For a batch of size 16,
                          'dino_frame_embeddings' would have a shape of [16, 16, 384],
                          'vggish_embedding' would be [16, 128],
                          'title' would be a list of 16 strings, etc.

        Returns:
            A tensor of embeddings with shape [batch_size, embedding_dim].
        """        
        # 1. Process Visual Features
        visual_features = self.visual_module(
            frames=batch_data['frames_for_low_level'],
            precomputed_dino_embeddings=batch_data['dino_frame_embeddings']
        )
        
        # 2. Process Audio Features
        audio_features = None
        if self.audio_module is not None:
            audio_features = self.audio_module(
                precomputed_vggish_embedding=batch_data.get('vggish_embedding'),
                waveform_for_spec_cnn=batch_data.get('waveform_for_spec_cnn')
            )
        
        # 3. Process Text Features
        text_features = None
        if self.text_module is not None:
            text_features = self.text_module(
                title=batch_data.get('title'), 
                plot=batch_data.get('plot'),
                tfidf_features=batch_data.get('tfidf_features'),
                year=batch_data.get('year'),
                language_idx=batch_data.get('language_idx')
            )
            
        # 4. Fuse all features into the final embedding
        raw_embedding = self.fusion_network(visual_features, audio_features, text_features)
        
        # Apply the final L2 normalization ->guarantees the input to loss function is always normalized.
        final_normalized_embedding = F.normalize(raw_embedding, p=2, dim=1, eps=1e-6)
        
        return final_normalized_embedding