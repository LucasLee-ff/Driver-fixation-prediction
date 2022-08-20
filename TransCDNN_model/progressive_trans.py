from vit_pytorch.vit import ViT2
from .vit_cdnn_modeling import DecoderBlock, SegmentationHead
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape=(8, 8)):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = self.shape[0], self.shape[1]
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return x


class ProgressiveTrans(nn.Module):
    def __init__(self, img_size=256):
        super(ProgressiveTrans, self).__init__()
        self.trans1 = ViT2(image_size=img_size, patch_size=8, dim=192, depth=4, heads=16, mlp_dim=2048, dropout=0.1
                           , emb_dropout=0.1)
        self.trans2 = ViT2(image_size=img_size, patch_size=16, dim=768, depth=4, heads=16, mlp_dim=2048, dropout=0.1
                           , emb_dropout=0.1)
        self.trans3 = ViT2(image_size=img_size, patch_size=32, dim=3072, depth=4, heads=16, mlp_dim=2048, dropout=0.1
                           , emb_dropout=0.1)

        self.reshape3 = Reshape((8, 8))
        self.reshape2 = Reshape((16, 16))
        self.reshape1 = Reshape((32, 32))

        self.decoder3 = DecoderBlock(3072, 768, 768)
        self.decoder2 = DecoderBlock(768, 192, 192)
        self.decoder1 = DecoderBlock(192, 96)
        self.decoder0 = DecoderBlock(96, 48)
        self.decoder = DecoderBlock(48, 24)

        self.segmentation_head = SegmentationHead(in_channels=24, out_channels=1, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_8 = self.trans1(x)
        x_8 = self.reshape1(x_8)

        x_16 = self.trans2(x)
        x_16 = self.reshape2(x_16)

        x_32 = self.trans3(x)
        x_32 = self.reshape3(x_32)

        y = self.decoder3(x_32, skip=x_16)
        y = self.decoder2(y, skip=x_8)
        y = self.decoder1(y)
        y = self.decoder0(y)
        y = self.decoder(y)

        y = self.segmentation_head(y)
        logits = self.sigmoid(y)
        return logits
