import torch
from torch import nn




class EmbeddingNetwork(nn.Module):
    def __init__(self, backbone, targets, in_channels, batch_size, device):
        super().__init__()
        self.backbone = backbone
        self.targets = targets
        self.batch_size = batch_size
        self.device = device
        self.in_channels = in_channels
        self.attention_modules = nn.ModuleList([AttentionModule(target, in_channels) for target in targets]) #TODO: compléter les tailles

    def forward(self, x):
        out = self.backbone(x)

        clf, emb, att_map = self.attention_modules[0](out)
        clfs = clf.unsqueeze(0)
        embs = emb.unsqueeze(0)
        attn_maps = att_map.unsqueeze(0)

        for idx, att_mod in enumerate(self.attention_modules[1:]):
            clf, emb, att_map = att_mod(out)
            clfs = torch.cat((clfs, clf.unsqueeze(0)), dim=0)
            embs = torch.cat((embs, emb.unsqueeze(0)), dim=0)
            attn_maps = torch.cat((attn_maps, att_map.unsqueeze(0)), dim=0)
        return clfs, embs, attn_maps


    def attn_to(self, device):
        for attn in self.attention_modules:
            attn.to(device)


class AttentionModule(nn.Module):
    def __init__(self, target, in_channels) -> None:
        super().__init__()
        self.target = target
        self.attention = nn.Conv2d(in_channels, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_map = self.attention(x)
        attention_map = torch.sigmoid(attention_map)
        #print(attention_map, attention_map.shape)
        clf = self.max_pool(attention_map)

        clf = clf.squeeze()
        if x.shape[0] == 1:
            clf = clf.unsqueeze(0)

        attentioned_features = attention_map * x
        embedding = self.avg_pool(attentioned_features)

        embedding = embedding.squeeze()
        if x.shape[0] == 1:
            embedding = embedding.unsqueeze(0)

        attention_map = attention_map.squeeze()
        if x.shape[0] == 1:
            attention_map = attention_map.unsqueeze(0)
        return clf, embedding, attention_map


class FreezeEmbeddingNetwork(nn.Module):
    def __init__(self, embedding_network, out_channels=512, freeze=True, twoloss=False):
        super().__init__()
        in_channels = embedding_network.in_channels*len(embedding_network.targets)+len(embedding_network.targets)

        self.freeze = freeze
        self.twoloss = twoloss
        self.batch_size = embedding_network.batch_size

        self.embedding_network = embedding_network
        self.fc1 = nn.Linear(in_channels, out_channels)

        if freeze:
            for param in self.embedding_network.parameters():
                param.requres_grad = False


    def forward(self, x):
        # Backbone + Attention
        if self.freeze:
            with torch.no_grad():
                out, embs, att_map = self.embedding_network(x)
                out = out.transpose(0, 1)
                embs = embs.transpose(0, 1)

                # FUSION
                embs = embs.reshape(embs.shape[0], -1)
                all_embs = torch.cat((out, embs), 1)

        else:
            out, embs, att_map = self.embedding_network(x)
            out = out.transpose(0, 1)
            embs = embs.transpose(0, 1)

            # FUSION
            embs = embs.reshape(embs.shape[0], -1)
            all_embs = torch.cat((out, embs), 1)


        # FC
        final_emb = self.fc1(all_embs)

        if self.twoloss:
            return out, final_emb
        else:
            return final_emb
