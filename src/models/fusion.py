#########################################################################
#================HYBRIDFUSIONNETWORK================================
#########################################################################

##REVISED HYBRID MODULE
class HybridFusionNetwork(nn.Module):
    def __init__(self, visual_dim, audio_dim, text_dim, embedding_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # --- Cross-Attention (Early Fusion) ---
        # This part models interactions between modalities
        self.visual_query = nn.Linear(visual_dim, visual_dim)
        
        # Audio influences Visual
        self.audio_kv = nn.Linear(audio_dim, visual_dim * 2) # Key and Value from Audio
        self.va_attention = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.va_norm = nn.LayerNorm(visual_dim)
        
        # Text influences Visual
        self.text_kv = nn.Linear(text_dim, visual_dim * 2) # Key and Value from Text
        self.vt_attention = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.vt_norm = nn.LayerNorm(visual_dim)
        
        # --- Late Fusion MLP ---
        # This part combines the individually processed and the cross-attended features
        
        #will have the original visual, the visual-after-audio-attention, and visual-after-text-attention
        late_fusion_input_dim = visual_dim * 3 
        
        self.late_fusion_mlp = nn.Sequential(
            nn.LayerNorm(late_fusion_input_dim),
            nn.Linear(late_fusion_input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        self._initialize_weights()
        print(f"🔧 Revised HybridFusionNetwork initialized.")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, visual_features, audio_features, text_features):
        # All inputs should be pre-normalized from the sub-modules
        
        # --- Early Fusion Part (Cross-Attention) ---
        # Visual features will act as the 'query' that asks questions of other modalities
        q = self.visual_query(visual_features).unsqueeze(1) # (B, 1, D_vis)

        # 1. Audio influences Visual
        audio_k, audio_v = self.audio_kv(audio_features).chunk(2, dim=-1)
        audio_k = audio_k.unsqueeze(1) # (B, 1, D_vis)
        audio_v = audio_v.unsqueeze(1) # (B, 1, D_vis)
        
        # Attention: Visual asks "what's in the audio?"
        va_out, _ = self.va_attention(query=q, key=audio_k, value=audio_v)
        va_out = va_out.squeeze(1) # (B, D_vis)
        visual_after_audio = self.va_norm(visual_features + va_out) 

        # 2. Text influences Visual
        text_k, text_v = self.text_kv(text_features).chunk(2, dim=-1)
        text_k = text_k.unsqueeze(1) # (B, 1, D_vis)
        text_v = text_v.unsqueeze(1) # (B, 1, D_vis)

        # Attention: Visual asks "what's in the text?"
        vt_out, _ = self.vt_attention(query=q, key=text_k, value=text_v)
        vt_out = vt_out.squeeze(1) # (B, D_vis)
        visual_after_text = self.vt_norm(visual_features + vt_out) 

        # --- Late Fusion Part ---
        # Combine the original visual with the attention-modified versions
        late_fusion_input = torch.cat([visual_features, visual_after_audio, visual_after_text], dim=1)
        
        final_embedding = self.late_fusion_mlp(late_fusion_input)

        # Final normalization before the loss function
        return F.normalize(final_embedding, p=2, dim=1, eps=1e-8)

#########################################################################
#================SIMPLEFUSIONNETWORK================================
#########################################################################

#SIMPLIFIED!
class SimpleFusionNetwork(nn.Module):
    def __init__(self, visual_dim, audio_dim, text_dim, embedding_dim=128, ablation_mode='full'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ablation_mode = ablation_mode

        self.fusion_input_dim = 0
        if visual_dim > 0: self.fusion_input_dim += visual_dim
        if (ablation_mode == 'visual_audio' or ablation_mode == 'full') and audio_dim > 0:
            self.fusion_input_dim += audio_dim
        if ablation_mode == 'full' and text_dim > 0:
            self.fusion_input_dim += text_dim
        
        if self.fusion_input_dim == 0:
            raise ValueError("Fusion input dimension cannot be zero.")

        print(f"Fusion Input Dim: {self.fusion_input_dim}, Final Embedding Dim: {embedding_dim}")

        intermediate_dim = max(embedding_dim * 2, self.fusion_input_dim // 2)
        intermediate_dim = min(intermediate_dim, 1024) # Cap the size

        # A more stable MLP structure
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(self.fusion_input_dim),
            nn.Linear(self.fusion_input_dim, intermediate_dim),
            nn.ReLU(),  # for stability
            nn.Dropout(0.5), # Heavier dropout
            nn.Linear(intermediate_dim, embedding_dim)
        )
        
        self._initialize_weights()
        print(f"🔧 Robust SimpleFusionNetwork (LayerNorm -> Linear -> ReLU -> Dropout -> Linear) initialized.")

    def _initialize_weights(self):
        for module in self.fusion_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.7)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, visual_features, audio_features=None, text_features=None):
        features_to_fuse = []
        if visual_features is not None: features_to_fuse.append(visual_features)
        
        if (self.ablation_mode == 'visual_audio' or self.ablation_mode == 'full') and audio_features is not None:
            features_to_fuse.append(audio_features)
            
        if self.ablation_mode == 'full' and text_features is not None:
            features_to_fuse.append(text_features)
        
        if not features_to_fuse:
            return torch.zeros((1, self.embedding_dim), device=visual_features.device if visual_features is not None else 'cpu')

        fused_input = torch.cat(features_to_fuse, dim=1)
        
        # Check for NaN/Inf BEFORE the MLP
        if torch.isnan(fused_input).any() or torch.isinf(fused_input).any():
            print("F- ❌ NaN/Inf detected in fused_input before MLP. Clamping.")
            fused_input = torch.nan_to_num(fused_input, nan=0.0, posinf=1.0, neginf=-1.0)
            
        embedding = self.fusion_mlp(fused_input)
        
        # Final normalization ->l for contrastive loss
        return F.normalize(embedding, p=2, dim=1, eps=1e-8)

#########################################################################
#================ULTRASIMPLEFUSIONNETWORK================================
#########################################################################

class UltraSimpleFusionNetwork(nn.Module):
    def __init__(self, visual_dim, audio_dim, text_dim, embedding_dim=128, ablation_mode='full'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ablation_mode = ablation_mode

        self.fusion_input_dim = 0
        if visual_dim > 0: self.fusion_input_dim += visual_dim
        if (ablation_mode == 'visual_audio' or ablation_mode == 'full') and audio_dim > 0:
            self.fusion_input_dim += audio_dim
        if ablation_mode == 'full' and text_dim > 0:
            self.fusion_input_dim += text_dim
        
        if self.fusion_input_dim == 0:
            raise ValueError("Fusion input dimension cannot be zero.")

        print(f"Fusion Input Dim: {self.fusion_input_dim}, Final Embedding Dim: {embedding_dim}")

        # --- ULTRA-STABLE MLP ---
        # A direct, normalized projection with no hidden layers.
        # This is the most stable architecture possible.
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(self.fusion_input_dim),
            nn.Linear(self.fusion_input_dim, self.embedding_dim)
        )
        
        self._initialize_weights()
        print(f"🔧 ULTRA-STABLE UltraSimpleFusionNetwork (LayerNorm -> Linear) initialized.")

    def _initialize_weights(self):
        for module in self.fusion_mlp.modules():
            if isinstance(module, nn.Linear):
                # Small gain for stability
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, visual_features, audio_features=None, text_features=None):
        features_to_fuse = []
        if visual_features is not None: features_to_fuse.append(visual_features)
        
        if (self.ablation_mode == 'visual_audio' or self.ablation_mode == 'full') and audio_features is not None:
            features_to_fuse.append(audio_features)
            
        if self.ablation_mode == 'full' and text_features is not None:
            features_to_fuse.append(text_features)
        
        if not features_to_fuse:
            return torch.zeros((1, self.embedding_dim), device='cpu')

        fused_input = torch.cat(features_to_fuse, dim=1)
        
        if torch.isnan(fused_input).any() or torch.isinf(fused_input).any():
            fused_input = torch.nan_to_num(fused_input, nan=0.0)
            
        embedding = self.fusion_mlp(fused_input)
        
        return F.normalize(embedding, p=2, dim=1, eps=1e-8)