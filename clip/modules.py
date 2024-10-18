

# file containing the image encoder, text encoder and projection head classes
# intuition and some implementation from https://github.com/moein-shariatnia/OpenAI-CLIP
# Shariatnia, M. M. (2021). Simple CLIP (Version 1.0.0) [Computer software]. https://doi.org/10.5281/zenodo.6845731

import torch
from torch import nn, TensorType
from data_handling import clip_data_prep
from data_handling import common
#import config as CFG
#import timm


class ImageEncoder(nn.Module):
    '''
    Othello board is represented as a [2,8,8] maxtrix, 8x8 1-0 values for black and white respectively
    based lightly on Chessclip and OpenClip implementation
    '''

    def __init__(self):
        super().__init__()

        inplanes = 3
        planes = 32
        stride = 1

        # padding set to 2 not to lose the important corner stones
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=2, bias=False) # 3x3 conv with stride 1
        self.bn1 = nn.BatchNorm2d(planes)
        self.ac1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes * self.expansion, 3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.stride = stride

    #def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        '''
        not using residual blocks        
        
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        out = gamma * out + beta
        
        out += identity
        '''

        out = self.act2(out)
        return out

class TextEncoder(nn.Module):
    '''
    Encoder for game moves (and only moves as it does not use i.e. Bert or DistilBert).
    Using an overly simple encoding as there are 64 possible moves (or 60 in practice),
    simply converting to a vector with size 64
    '''

    def __init__(self):
        super().__init__()
        self.ranks = 'ABCDEFGH'

    def forward(self, move):

        board_array = [0] * 64

        if move != clip_data_prep.END_OF_GAME:
            row, col = common.get_move_coords(move)
            value = row*8 + col

            # setting the only value if not end of game
            board_array[value] = 1

        # return a torch tensor with size 64
        return torch.FloatTensor(board_array)

class ProjectionHead(nn.Module):
    '''
    Projection head for CLIP training
    based on Shariatnia, M. M. (2021). Simple CLIP (Version 1.0.0) [Computer software]. https://doi.org/10.5281/zenodo.6845731
    '''

    def __init__(self, embedding_dim, projection_dim = 64, dropout = 0.1):
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x