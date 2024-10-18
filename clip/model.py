

from modules import ImageEncoder, TextEncoder, ProjectionHead
import torch
from torch import nn, TensorType
import torch.nn.functional as F

class CLIPModel(nn.Module):
    '''
    Main model for CLIP training othello dataset
    '''

    def __init__(self):

        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead()
        self.text_projection = ProjectionHead()
        self.temperature = 0.1

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["boards"])
        text_features = self.text_encoder(
            input_ids=batch["moves"])

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


# test method

if __name__ == '__main__':
    images = torch.randn(3, 2, 8, 8)
    #input_ids = torch.randint(5, 300, size=(8, 25))
    input_moves = ['A2','B3','H8']
    batch = {
        'image': images,
        'input_ids': input_moves
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")