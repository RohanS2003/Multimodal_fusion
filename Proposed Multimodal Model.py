def multimodal_fusion(face_embedding, gait_embedding, face_weight=0.7, gait_weight=0.3):
    # Normalize embeddings
    face_embedding = F.normalize(face_embedding, p=2, dim=1)
    gait_embedding = F.normalize(gait_embedding, p=2, dim=1)
    
    # Weighted Similarity Score
    similarity_score = face_weight * (face_embedding @ face_embedding.T) + \
                       gait_weight * (gait_embedding @ gait_embedding.T)
    return similarity_score
