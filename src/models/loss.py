#########################################################################
#==========================PAIR-COSINE LOSS==============================
#########################################################################

class ContrastiveLossCosine(nn.Module):
    def __init__(self, margin=0.5): # A smaller margin is often used for Cosine distance
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin
        self.eps = 1e-9 # For numerical stability

    def forward(self, embedding1, embedding2, label):
        cosine_similarity = F.cosine_similarity(embedding1, embedding2, dim=1, eps=self.eps)

        # Distance = 1 - Similarity
        cosine_distance = 1 - cosine_similarity

        loss_positive = (label) * torch.pow(cosine_distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
        
        loss_contrastive = torch.mean(loss_positive + loss_negative)

        return loss_contrastive


#########################################################################
#==========================PAIR-EUCLIDEAN LOSS==============================
#########################################################################
class ContrastiveLossEuclidean(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLossEuclidean, self).__init__()
        self.margin = margin
        self.eps = 1e-9 # for stability in sqrt

    def forward(self, embedding1, embedding2, label):
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, p=2, eps=self.eps)

        loss_positive = (label) * torch.pow(euclidean_distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean(loss_positive + loss_negative)
        
        return loss_contrastive

#########################################################################
#==========================TRIPLET-COSINE LOSS==============================
#########################################################################
class TripletLossCosine(nn.Module):
    """
    Triplet loss function for cosine similarity.
    The goal is to make the anchor more similar to the positive than to the negative,
    by at least a certain margin.
    
    Objective: sim(a, p) > sim(a, n) + margin
    Loss: max(0, sim(a, n) - sim(a, p) + margin)
    """
    def __init__(self, margin=0.3):
        super(TripletLossCosine, self).__init__()
        self.margin = margin
        self.eps = 1e-9 # For numerical stability

    def forward(self, anchor_emb, positive_emb, negative_emb):
        # All embeddings should be L2-normalized from the model
        sim_pos = F.cosine_similarity(anchor_emb, positive_emb, dim=1, eps=self.eps)
        sim_neg = F.cosine_similarity(anchor_emb, negative_emb, dim=1, eps=self.eps)
        
        loss = torch.clamp(sim_neg - sim_pos + self.margin, min=0.0)
        
        return torch.mean(loss)

#########################################################################
#==========================TRIPLET-EUCLIDEAN LOSS==============================
#########################################################################
class TripletLossEuclidean(nn.Module):
    """
    Triplet loss function for Euclidean distance.
    The goal is to make the anchor's distance to the positive smaller than its
    distance to the negative, by at least a certain margin.
    
    Objective: dist(a, p) + margin < dist(a, n)
    Loss: max(0, dist(a, p) - dist(a, n) + margin)
    """
    def __init__(self, margin=1.0):
        super(TripletLossEuclidean, self).__init__()
        self.margin = margin

    def forward(self, anchor_emb, positive_emb, negative_emb):
        dist_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        dist_neg = F.pairwise_distance(anchor_emb, negative_emb, p=2)
        
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        
        return torch.mean(loss)
