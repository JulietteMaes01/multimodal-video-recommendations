#########################################################################
#===========================VISUAL MODULE================================
#########################################################################

class VisualProcessingModule(nn.Module):
    def __init__(self, backbone="dinov2", low_level_features=True, use_precomputed_embeddings=False, dinov2_embedding_dim=384):
        """
        Visual Processing Module that can work with either:
        1. Raw frames + backbone processing (original mode)
        2. Precomputed DINOv2 embeddings (new mode)
        
        Args:
            backbone: "dinov2" or "resnet" - only used when use_precomputed_embeddings=False
            low_level_features: Whether to extract and use low-level visual features
            use_precomputed_embeddings: If True, expects precomputed DINOv2 embeddings as input
            dinov2_embedding_dim: Dimension of DINOv2 embeddings (384 for ViT-S/14)
        """
        super().__init__()
        self.backbone_type = backbone
        self.low_level_features = low_level_features
        self.use_precomputed_embeddings = use_precomputed_embeddings

        # Setup backbone and feature dimensions
        if use_precomputed_embeddings:
            # No backbone needed, we'll receive precomputed embeddings
            self.backbone = None
            self.feature_dim = dinov2_embedding_dim
            print(f"Using precomputed embeddings mode with feature_dim={self.feature_dim}")
        else:
            # Original mode - setup backbone
            if backbone == "dinov2":
                self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
                self.feature_dim = 384
                
                # Conservative fine-tuning: Only unfreeze the final norm layer
                print("DINOv2 Fine-tuning: Conservative - ONLY final 'norm' layer.")
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False  # Freeze all first
                    
                if hasattr(self.backbone, 'norm') and isinstance(self.backbone.norm, nn.LayerNorm):
                    print("Unfreezing DINOv2's final 'norm' layer.")
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True
                    
                    num_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
                    print(f"DINOv2 trainable parameters: {num_trainable}")
                else:
                    print("DINOv2 'norm' layer not found. Backbone remains fully frozen.")
                    
            elif backbone == "resnet":
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
                self.feature_dim = 2048
                
                # Freeze all but last few layers
                for param in list(self.backbone.parameters())[:-10]:
                    param.requires_grad = False

        # MLP layers for processing features
        self.norm_after_backbone_pooling = nn.LayerNorm(self.feature_dim, eps=1e-5)
        self.dropout_before_fusion = nn.Dropout(0.4)
        
        self.frame_fusion_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 2)
        )
        
        self.dropout_after_fusion = nn.Dropout(0.4)

        # Low-level feature processing
        if low_level_features:
            self.low_level_dim_output = 64
            self.low_level_projection = nn.Linear(5, self.low_level_dim_output)
            self.output_dim = (self.feature_dim // 2) + self.low_level_dim_output
        else:
            self.output_dim = self.feature_dim // 2

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with conservative gains to prevent gradient explosion."""
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'frame_fusion_mlp.0' in module_name:
                    gain = 1.0
                    print(f"Applying standard gain ({gain}) to {module_name}")
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                else:
                    gain = nn.init.calculate_gain('relu')
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                    
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, frames, precomputed_dino_embeddings=None):
        """
        Forward pass supporting both modes:
        
        Mode 1 (original): forward(frames)
        - frames: (Batch, NumFrames, C, H, W)
        
        Mode 2 (precomputed): forward(frames, precomputed_dino_embeddings)
        - frames: (Batch, NumFrames, C, H, W) - used only for low-level features
        - precomputed_dino_embeddings: (Batch, NumFrames, DINO_DIM)
        """
        
        if self.use_precomputed_embeddings:
            if precomputed_dino_embeddings is None:
                raise ValueError("precomputed_dino_embeddings must be provided when use_precomputed_embeddings=True")
            return self._forward_with_precomputed(frames, precomputed_dino_embeddings)
        else:
            return self._forward_with_backbone(frames)

    def _forward_with_precomputed(self, frames_for_low_level, precomputed_dino_frame_embeddings):
        """Forward pass using precomputed DINOv2 embeddings."""
        
        # 1. Pool the precomputed per-frame embeddings
        video_features_pooled = precomputed_dino_frame_embeddings.mean(dim=1)  # (Batch, DINO_DIM)
        
        # --- ADDED L2 NORMALIZATION FOR POOLED DINO FEATURES ---
        if torch.isnan(video_features_pooled).any() or torch.isinf(video_features_pooled).any():
            print("V- Pooled DINO had NaN/Inf before L2 norm. Clamping.")
            video_features_pooled = torch.nan_to_num(video_features_pooled, nan=0.0, posinf=1.0, neginf=-1.0) # Added clamping range
        
        # Stabilize before F.normalize
        pooled_dino_norm_val = torch.norm(video_features_pooled, p=2, dim=1, keepdim=True)
        # Add a small epsilon to prevent division by zero if norm is exactly zero after nan_to_num
        if torch.any(pooled_dino_norm_val < 1e-7): 
            noise = torch.randn_like(video_features_pooled) * 1e-7
            # Apply noise only where norm is too small
            video_features_pooled = torch.where(
                (pooled_dino_norm_val < 1e-7).expand_as(video_features_pooled), 
                video_features_pooled + noise, 
                video_features_pooled
            )
            
        video_features_pooled_for_norm = video_features_pooled
        if torch.any(torch.norm(video_features_pooled_for_norm, p=2, dim=1) < 1e-7): # Check again
             # If still zero norm for some (e.g. all-zero input + all-zero noise), make them tiny non-zero
             video_features_pooled_for_norm = video_features_pooled_for_norm + 1e-7 * torch.ones_like(video_features_pooled_for_norm)


        video_features_pooled_normalized_l2 = F.normalize(video_features_pooled_for_norm, p=2, dim=1, eps=1e-6)

        # 2. Normalization and MLP processing
        video_features_normed_by_layernorm = self.norm_after_backbone_pooling(video_features_pooled_normalized_l2) # Pass the L2-normalized version
        
        if torch.isnan(video_features_normed_by_layernorm).any() or torch.isinf(video_features_normed_by_layernorm).any():
            video_features_normed_by_layernorm = torch.nan_to_num(video_features_normed_by_layernorm, nan=0.0)

        video_features_to_fuse = self.dropout_before_fusion(video_features_normed_by_layernorm)
        
        # Check BatchNorm variance if needed
        if video_features_to_fuse.size(0) > 1: # Ensure batch size > 1 for variance calculation
            input_to_bn = self.frame_fusion_mlp[0](video_features_to_fuse)
            # Ensure input_to_bn is not empty and has variance before checking
            if input_to_bn.numel() > 0 and input_to_bn.size(0) > 1: 
                var_bn_input_check = input_to_bn.var(dim=0, unbiased=False)
                if (var_bn_input_check < 1e-6).any():
                    problematic_channels = (var_bn_input_check < 1e-6).sum().item()
                    print(f"V- ⚠️ Low variance in BatchNorm1d input: {problematic_channels}/{input_to_bn.shape[1]} channels (after DINO L2 + LayerNorm)")

        fused_video_features = self.frame_fusion_mlp(video_features_to_fuse)
        if torch.isnan(fused_video_features).any() or torch.isinf(fused_video_features).any():
            fused_video_features = torch.nan_to_num(fused_video_features, nan=0.0)

        final_high_level_features = self.dropout_after_fusion(fused_video_features)

        # 3. Low-level features (calculated on-the-fly using original frames)
        if self.low_level_features and frames_for_low_level is not None:
            low_level_raw_avg = self.extract_low_level_features(frames_for_low_level)  # (Batch, 5)
            
            if torch.isnan(low_level_raw_avg).any():
                low_level_raw_avg = torch.nan_to_num(low_level_raw_avg, nan=0.0)
            
            low_level_projected = self.low_level_projection(low_level_raw_avg)
            low_level_projected = torch.tanh(low_level_projected)
            if torch.isnan(low_level_projected).any():
                low_level_projected = torch.nan_to_num(low_level_projected, nan=0.0)

            output_features = torch.cat([final_high_level_features, low_level_projected], dim=1)
        else:
            output_features = final_high_level_features
        
        if torch.isnan(output_features).any():
            output_features = torch.nan_to_num(output_features, nan=0.0)
        
        output_magnitude_visual = torch.norm(output_features, p=2, dim=1, keepdim=True)
        if torch.any(output_magnitude_visual < 1e-7):
            noise_visual = torch.randn_like(output_features) * 1e-7
            output_features = torch.where((output_magnitude_visual < 1e-7).expand_as(output_features), output_features + noise_visual, output_features)
            # print(f"V- Applied noise to final visual features with near-zero norm before F.normalize.")

        output_features_final_normalized = F.normalize(output_features, p=2, dim=1, eps=1e-6)
        
        # print(f"V- Post-FinalNorm: mean={output_features_final_normalized.mean().item():.4f}, std={output_features_final_normalized.std().item():.4f}, norm={torch.norm(output_features_final_normalized, p=2, dim=1).mean().item():.4f}")

        if torch.isnan(output_features_final_normalized).any():
            print("V- 👺 NaN in visual_module final output AFTER F.normalize. Zeroing.")
            output_features_final_normalized = torch.zeros_like(output_features_final_normalized)
            
        return output_features_final_normalized

    def _forward_with_backbone(self, frames):
        """Original forward pass using backbone feature extraction."""
        batch_size, num_frames = frames.shape[0], frames.shape[1]

        if torch.isnan(frames).any():
            print("V- WARNING: Input frames contain NaN values! Replacing with zeros.")
            frames = torch.nan_to_num(frames, nan=0.0)

        # 1. Backbone Feature Extraction
        # Handle partial freezing for DINOv2
        if self.backbone_type == "dinov2":
            # Store original requires_grad states
            original_requires_grad = {}
            for name, param in self.backbone.named_parameters():
                original_requires_grad[name] = param.requires_grad
                # Temporarily freeze parts that should be frozen during forward pass
                if 'norm' not in name:
                    param.requires_grad_(False)
        
        frames_flat = frames.view(-1, 3, frames.shape[3], frames.shape[4])
        chunk_size = 8  # Adjust based on GPU memory
        frame_features_list = []
        
        for i in range(0, frames_flat.size(0), chunk_size):
            chunk = frames_flat[i:i+chunk_size]
            chunk_features = self.backbone(chunk)
            frame_features_list.append(chunk_features)
        
        frame_features_flat = torch.cat(frame_features_list, dim=0)

        # Restore original requires_grad states for DINOv2
        if self.backbone_type == "dinov2":
            for name, param in self.backbone.named_parameters():
                param.requires_grad_(original_requires_grad[name])

        if torch.isnan(frame_features_flat).any() or torch.isinf(frame_features_flat).any():
            print("V- ❌ CRITICAL: NaN/Inf detected from backbone output! Cleaning.")
            frame_features_flat = torch.nan_to_num(frame_features_flat, nan=0.0, posinf=10.0, neginf=-10.0)
        
        frame_features = frame_features_flat.view(batch_size, num_frames, -1)
        frame_features = torch.clamp(frame_features, -10.0, 10.0)  # Safety clamp

        # 2. Temporal Pooling and Normalization
        video_features_pooled = frame_features.mean(dim=1)

        if torch.isnan(video_features_pooled).any() or torch.isinf(video_features_pooled).any():
            print("V- ❌ NaN/Inf in video_features_pooled! Zeroing.")
            video_features_pooled = torch.nan_to_num(video_features_pooled, nan=0.0)

        video_features_normed = self.norm_after_backbone_pooling(video_features_pooled)
        if torch.isnan(video_features_normed).any() or torch.isinf(video_features_normed).any():
            print("V- ❌ NaN/Inf after normalization! Zeroing.")
            video_features_normed = torch.nan_to_num(video_features_normed, nan=0.0)

        # 3. Frame Fusion MLP
        video_features_to_fuse = self.dropout_before_fusion(video_features_normed)

        fused_video_features = self.frame_fusion_mlp(video_features_to_fuse)

        # 4. Final Dropout
        final_high_level_features = self.dropout_after_fusion(fused_video_features)

        # 5. Low-Level Features (if enabled)
        if self.low_level_features:
            low_level_raw = self.extract_low_level_features(frames)
            if torch.isnan(low_level_raw).any() or torch.isinf(low_level_raw).any():
                print("V- ❌ NaN/Inf in low_level_raw! Zeroing.")
                low_level_raw = torch.nan_to_num(low_level_raw, nan=0.0)
            
            low_level_projected = self.low_level_projection(low_level_raw)
            low_level_projected = torch.tanh(low_level_projected)  # Bound low-level features
            if torch.isnan(low_level_projected).any() or torch.isinf(low_level_projected).any():
                print("V- ❌ NaN/Inf in low_level_projected! Zeroing.")
                low_level_projected = torch.nan_to_num(low_level_projected, nan=0.0)

            output_features = torch.cat([final_high_level_features, low_level_projected], dim=1)
        else:
            output_features = final_high_level_features
        
        if torch.isnan(output_features).any() or torch.isinf(output_features).any():
            print("V- ❌ NaN/Inf in final output_features! Zeroing.")
            output_features = torch.nan_to_num(output_features, nan=0.0)
            
        return output_features

    def extract_low_level_features(self, frames):
        """Extract low-level visual features like brightness, contrast, edge density."""
        B, N, C, H, W = frames.shape
        frames_flat = frames.view(B * N, C, H, W)

        # Convert to grayscale robustly
        if frames_flat.dtype == torch.uint8:
            frames_flat_float = frames_flat.float() / 255.0
        else:
            frames_flat_float = frames_flat

        gray_frames_flat = (0.299 * frames_flat_float[:, 0:1] + 
                           0.587 * frames_flat_float[:, 1:2] + 
                           0.114 * frames_flat_float[:, 2:3])
        
        brightness = gray_frames_flat.mean(dim=[1, 2, 3])
        contrast = gray_frames_flat.std(dim=[1, 2, 3], unbiased=False) + 1e-8

        # Edge density using Kornia if available
        edge_density = torch.zeros_like(brightness)
        if KFilters is not None:
            try:
                sobel_output = KFilters.sobel(gray_frames_flat)
                if sobel_output.shape[1] == 2:  # Returns gx and gy
                    sobel_magnitude = torch.sqrt(sobel_output[:,0:1]**2 + sobel_output[:,1:2]**2 + 1e-10)
                else:  # Returns magnitude
                    sobel_magnitude = sobel_output
                edge_density = sobel_magnitude.mean(dim=[1, 2, 3])
            except Exception as e:
                print(f"V- Kornia Sobel filter error: {e}. Using zero for edge density.")
                edge_density = torch.zeros_like(brightness)
        
        # Placeholders for additional features
        color_entropy = torch.zeros_like(brightness)
        rule_of_thirds_metric = torch.zeros_like(brightness)

        low_level_features_flat = torch.stack([
            brightness, contrast, edge_density, color_entropy, rule_of_thirds_metric
        ], dim=1)
        
        low_level_features_video = low_level_features_flat.view(B, N, -1)
        features_avg = low_level_features_video.mean(dim=1)
        
        if torch.isnan(features_avg).any():
            features_avg = torch.nan_to_num(features_avg, nan=0.0)
            
        return features_avg


#########################################################################
#===========================AUDIO MODULE================================
#########################################################################

class OptimizedAudioProcessingModule(nn.Module):
    def __init__(self, use_vggish=True, use_spectrogram_cnn=True, vggish_embedding_dim=128):
        super().__init__()
        self.use_vggish = use_vggish
        self.use_spectrogram_cnn = use_spectrogram_cnn
        
        self.vggish_dim = vggish_embedding_dim if self.use_vggish else 0
            
        # Spectrogram CNN setup
        if self.use_spectrogram_cnn:
            self.spec_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),      # 0
                nn.BatchNorm2d(16),                             # 1
                nn.GELU(),                                      # 2
                nn.MaxPool2d(2),                                # 3
                nn.Dropout2d(0.1),                              # 4
                nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 5
                nn.BatchNorm2d(32),                             # 6
                nn.GELU(),                                      # 7
                nn.MaxPool2d(2),                                # 8
                nn.Dropout2d(0.1),                              # 9
                nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 10
                nn.BatchNorm2d(64),                             # 11
                nn.GELU(),                                      # 12
                nn.AdaptiveAvgPool2d((1, 1)),                   # 13
                nn.Dropout2d(0.2)                               # 14
            )
            
            self.spec_dim = 64
        else:
            self.spec_dim = 0

        # Fusion layer setup
        self.input_to_fusion_dim = self.vggish_dim + self.spec_dim 
        self.fused_audio_intermediate_dim = 128 
        self.final_audio_output_dim = 192 

        if self.input_to_fusion_dim > 0 and (self.vggish_dim == 0 or self.spec_dim == 0):
            # Only one feature type is active
            self.fusion = nn.Identity()
            self.output_dim = self.input_to_fusion_dim
            print(f"AUDIO_MODULE: Using Identity for self.fusion as only one audio feature type "
                  f"({'VGGish' if self.vggish_dim > 0 else 'SpecCNN'}) is active. "
                  f"Output dim for temporal pooling: {self.output_dim}")
        elif self.vggish_dim > 0 and self.spec_dim > 0:
            # Both feature types are active - need fusion
            self.fusion = nn.Sequential(
                nn.Linear(self.input_to_fusion_dim, self.fused_audio_intermediate_dim),
                nn.LayerNorm(self.fused_audio_intermediate_dim, eps=1e-5), 
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(self.fused_audio_intermediate_dim, self.final_audio_output_dim) 
            )
            self.output_dim = self.final_audio_output_dim 
        else:
            # No audio features active
            self.fusion = nn.Identity()
            self.output_dim = 0
            print("AUDIO_MODULE: No audio features active, self.fusion is Identity, output_dim is 0.")

        # Temporal pooling setup
        self.temporal_pooling = nn.Sequential(
            nn.Linear(self.output_dim if self.output_dim > 0 else 1, self.output_dim if self.output_dim > 0 else 1), 
            nn.Tanh()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module_name, module in self.named_modules(): 
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                if 'fusion.' in module_name or 'temporal_pooling.' in module_name:
                    gain_val = 0.5 
                    nn.init.xavier_uniform_(module.weight, gain=gain_val) 
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.5) 
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)): 
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
    def compute_spectrogram_batch(self, audio, sr=16000):
        """Compute spectrograms for a batch of audio samples."""
        device = audio.device
        batch_size = audio.shape[0]
        audio = torch.clamp(audio, -1.0, 1.0)
        window = torch.hann_window(512, device=device)
        specs = []
        
        for i in range(batch_size):
            audio_sample = audio[i]
            if audio_sample.std() > 1e-8: 
                audio_sample = (audio_sample - audio_sample.mean()) / (audio_sample.std() + 1e-8)
            stft = torch.stft(
                audio_sample, 
                n_fft=512, 
                hop_length=256, 
                window=window,
                return_complex=True
            )
            spec = torch.abs(stft)
            spec = torch.clamp(spec, min=1e-8, max=100.0) 
            spec = torch.log(spec + 1e-8)
            if spec.std() > 1e-8:
                spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            spec = torch.clamp(spec, -5.0, 5.0) 
            specs.append(spec)
            
        specs = torch.stack(specs).unsqueeze(1)
        return specs

    def forward(self, precomputed_vggish_embedding=None, waveform_for_spec_cnn=None):
        """
        Forward pass with separate inputs for VGGish embeddings and spectrogram CNN.
        
        Args:
            precomputed_vggish_embedding: (Batch, VGGISH_DIM) - precomputed VGGish features
            waveform_for_spec_cnn: (Batch, NumSamples) or (Batch, 1, NumSamples) - raw audio for spec CNN
        """
        features_to_combine = []
        
        # Process VGGish embeddings if provided
        if self.use_vggish and precomputed_vggish_embedding is not None:
            if torch.isnan(precomputed_vggish_embedding).any() or torch.isinf(precomputed_vggish_embedding).any():
                print("A- VGGish emb had NaN/Inf before L2 norm. Clamping.")
                precomputed_vggish_embedding = torch.nan_to_num(precomputed_vggish_embedding, nan=0.0)
            
            # --- ADD L2 NORMALIZATION FOR VGGISH EMBEDDING ---
            vggish_norm_val = torch.norm(precomputed_vggish_embedding, p=2, dim=1, keepdim=True)
            if torch.any(vggish_norm_val < 1e-7):
                noise = torch.randn_like(precomputed_vggish_embedding) * 1e-7
                precomputed_vggish_embedding = torch.where((vggish_norm_val < 1e-7).expand_as(precomputed_vggish_embedding),
                                                           precomputed_vggish_embedding + noise, precomputed_vggish_embedding)
            
            normalized_vggish_embedding = F.normalize(precomputed_vggish_embedding, p=2, dim=1, eps=1e-6)
            # print(f"A- VGGish (after L2 norm): mean={normalized_vggish_embedding.mean().item():.4f}, std={normalized_vggish_embedding.std().item():.4f}, norm={torch.norm(normalized_vggish_embedding, p=2, dim=1).mean().item():.4f}")
            features_to_combine.append(normalized_vggish_embedding)
            # --- END ADDED L2 NORMALIZATION ---
        
        # Process spectrogram CNN if waveform provided
        if self.use_spectrogram_cnn and waveform_for_spec_cnn is not None:
            # Ensure waveform is (Batch, NumSamples) for compute_spectrogram_batch
            if waveform_for_spec_cnn.dim() == 3 and waveform_for_spec_cnn.shape[1] == 1:
                waveform_for_spec_cnn = waveform_for_spec_cnn.squeeze(1)

            specs = self.compute_spectrogram_batch(waveform_for_spec_cnn)  # (Batch, 1, N_MELS, Width)
            
            # Handle dimension issues
            if specs.dim() == 5: 
                specs = specs.squeeze(2)
            specs = torch.clamp(specs, -5.0, 5.0)
            
            # Process through CNN
            spec_cnn_output = self.spec_cnn(specs).squeeze(-1).squeeze(-1)
            spec_cnn_output = torch.clamp(spec_cnn_output, -10.0, 10.0)
            if torch.isnan(spec_cnn_output).any(): 
                spec_cnn_output = torch.nan_to_num(spec_cnn_output, nan=0.0)
            features_to_combine.append(spec_cnn_output)
            
        # Handle case where no features are available
        if not features_to_combine:
            expected_output_dim = self.output_dim if self.output_dim > 0 else 1
            # Determine batch size from one of the inputs if possible, or default if all are None
            bs = (precomputed_vggish_embedding.shape[0] if precomputed_vggish_embedding is not None else 
                  (waveform_for_spec_cnn.shape[0] if waveform_for_spec_cnn is not None else 1))
            device = (precomputed_vggish_embedding.device if precomputed_vggish_embedding is not None else
                     (waveform_for_spec_cnn.device if waveform_for_spec_cnn is not None else torch.device('cpu')))
            return torch.zeros(bs, expected_output_dim, device=device)

        # Combine features
        if len(features_to_combine) > 1:
            combined_audio_features = torch.cat(features_to_combine, dim=1)
            processed_features = self.fusion(combined_audio_features)
        elif features_to_combine:
            processed_features = features_to_combine[0]
            # Apply fusion if it's not Identity
            if not isinstance(self.fusion, nn.Identity):
                processed_features = self.fusion(processed_features)

        # Apply temporal pooling and normalization
        output = self.temporal_pooling(processed_features)
        output = F.normalize(output, p=2, dim=1, eps=1e-8)
        return output

#########################################################################
#===========================TEXT MODULE================================
#########################################################################

class TextProcessingModule(nn.Module):
    def __init__(self, use_bert=True, use_tfidf=True, num_languages=150,
                 lang_embedding_dim=16):
        super().__init__()
        self.use_bert = use_bert
        self.use_tfidf = use_tfidf
        
        if use_bert:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert_dim = 768
            for layer in self.bert.encoder.layer:
                for param in layer.parameters():
                    param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = False
        else:
            self.bert_dim = 0
        
        if use_tfidf:
            self.tfidf_dim = 100
            self.tfidf_projection = nn.Linear(5000, self.tfidf_dim)
            # Changed eps to be more conservative for LayerNorm
            self.tfidf_norm = nn.LayerNorm(self.tfidf_dim, eps=1e-6)
        else:
            self.tfidf_dim = 0
            
        # --- NEW: Language and Year Setup ---
        self.lang_embedding_dim = lang_embedding_dim
        self.language_embedding = nn.Embedding(num_languages, self.lang_embedding_dim)
        self.year_dim = 1 # Year is a single feature
            
        # --- MODIFIED: Update the fusion input dimension ---
        self.input_dim = self.bert_dim + self.tfidf_dim + self.lang_embedding_dim + self.year_dim
        
        # Fusion MLP
        self.output_dim = 384
        self.fusion = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.output_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
    def process_bert(self, text_batch):
        encoded = self.tokenizer(text_batch, padding=True, truncation=True, 
                                max_length=128, return_tensors='pt')
        encoded = {k: v.to(next(self.bert.parameters()).device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.bert(**encoded)
            bert_features = outputs.last_hidden_state[:, 0, :]
        
        # Enhanced NaN/Inf checking for BERT features
        if torch.isnan(bert_features).any() or torch.isinf(bert_features).any():
            print("T- 👺 NaN/Inf in BERT features! Clamping aggressively.")
            bert_features = torch.nan_to_num(bert_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return bert_features
    
    def process_tfidf(self, tfidf_features):
        # Input sanitization
        if torch.isnan(tfidf_features).any() or torch.isinf(tfidf_features).any():
            tfidf_features = torch.nan_to_num(tfidf_features, nan=0.0, posinf=1.0, neginf=-1.0)

        projected = self.tfidf_projection(tfidf_features)
        
        # Post-projection sanitization
        if torch.isnan(projected).any() or torch.isinf(projected).any():
            projected = torch.nan_to_num(projected, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if self.training and projected.size(0) > 1:
            # Check for zero variance and add tiny noise if needed
            var = projected.var(dim=0, keepdim=True)
            zero_var_mask = var < 1e-8
            if zero_var_mask.any():
                noise = torch.randn_like(projected) * 1e-8
                projected = projected + noise

        normalized_tfidf = self.tfidf_norm(projected)
        
        if torch.isnan(normalized_tfidf).any() or torch.isinf(normalized_tfidf).any():
            normalized_tfidf = torch.nan_to_num(normalized_tfidf, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized_tfidf
        
    def forward(self, title, plot, tfidf_features=None, year=None, language_idx=None):      
        features = []
        text_batch = [f"{t} [SEP] {p}" for t, p in zip(title, plot)]
        
        if self.use_bert:
            bert_features = self.process_bert(text_batch)  # Already handles NaNs
            
            # L2 normalization for BERT features
            bert_norm_val = torch.norm(bert_features, p=2, dim=1, keepdim=True)
            if torch.any(bert_norm_val < 1e-7):
                noise = torch.randn_like(bert_features) * 1e-7
                bert_features = torch.where((bert_norm_val < 1e-7).expand_as(bert_features),
                                           bert_features + noise, bert_features)
            bert_features = F.normalize(bert_features, p=2, dim=1, eps=1e-6)
            features.append(bert_features)
            
        if self.use_tfidf and tfidf_features is not None:
            tfidf_projected = self.process_tfidf(tfidf_features)
            features.append(tfidf_projected)
        elif self.use_tfidf and tfidf_features is None:
            print("T- TF-IDF features were expected but not provided. Creating zeros tensor.")
            batch_size = len(title)
            device = next(self.parameters()).device
            tfidf_zeros = torch.zeros(batch_size, self.tfidf_dim, device=device)
            features.append(tfidf_zeros)

        if language_idx is not None:
            if language_idx.dim() > 1:
                language_idx = language_idx.squeeze(1)
            lang_emb = self.language_embedding(language_idx)
            features.append(lang_emb)

        if year is not None:
            features.append(year)

        if not features:
             print("T- WARNING: No text features were processed. Returning zeros.")
             batch_size = len(title) if title else 1
             device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
             return torch.zeros(batch_size, self.output_dim, device=device)

        combined = torch.cat(features, dim=1)
        
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
        
        output = self.fusion(combined)
        
        # Sanitize output from fusion MLP
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check for zero vectors before normalization and handle them
        output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
        
        # Add noise to samples with near-zero norm
        if torch.any(output_norm < 1e-7):
            noise = torch.randn_like(output) * 1e-7
            near_zero_mask = (output_norm < 1e-7).expand_as(output)
            output = torch.where(near_zero_mask, output + noise, output)
        
        # Apply F.normalize with increased eps
        output = F.normalize(output, p=2, dim=1, eps=1e-6)
        
        # Final check after F.normalize
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("T- 👺 NaN/Inf in text_module output AFTER F.normalize. Re-clamping to zeros.")
            output_norm_check = torch.norm(output, p=2, dim=1, keepdim=True)
            print(f"   Input norms to F.normalize: min={output_norm_check.min().item():.8f}, "
                  f"max={output_norm_check.max().item():.8f}, mean={output_norm_check.mean().item():.8f}")
            output = torch.zeros_like(output)
        
        return output