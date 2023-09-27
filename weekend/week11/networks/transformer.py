import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.Q = 
        self.K = 
        self.V = 
        self.Linear = nn.Linear()

    def forward(self, feature):
        return feature

class Encoder_Layer(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.MHAttention = 
        self.norm1 = 
        self.norm2 = 
        self.ff = nn.Linear()

    def forward(self, feature):
        res_feat = feature.copy()
        feature = self.MHAttention(feature)
        feature = feature + res_feat
        feature = self.norm1(feature)

        res_feat = feature.copy()
        feature = self.ff(feature)
        feature = feature + res_feat
        feature = self.norm2(feature)
        return feature

class Positional_Encoding(nn.Module):
    def __init__(self, N):
        super().__init__()

    def forward(self, text):
        return feature

class Trans_Encoder(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.word_emb
        self.pos_enc = Positional_Encoding()
        enc_layers = []

        for n in range(N):
            enc_layers.append(Encoder_Layer())
        self.enc_layers = nn.Sequential(*enc_layers)

    def forward(self, text):
        emb = self.word_emb(text)
        emb = emb + self.pos_enc(text)
        feature = self.enc_layers(emb)
        return feature
    
class Trans_Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, feature):
        out = self.fc(feature)
        return out
    
class IMDBTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_encoder = Trans_Encoder()
        self.classifier = Trans_Classifier()

    def forward(self, text, length):
        feature = self.trans_encoder(text)
        out = self.clasifier(feature)
        return out
    