import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from encoders import LanguageEmbeddingLayer, RNNEncoder, SubNet

from transformers import BertModel, BertConfig

class Fusion(nn.Module):
    def __init__(self, hp):
        
        # Base Encoders
        super().__init__()
        self.hp = hp
        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer()
        self.visual_enc = RNNEncoder(
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout

        # Trimodal Settings
        self.fusion_prj = SubNet(
            in_size=dim_sum,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
    
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)

        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)

        fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))

        return text, acoustic, visual, fusion, preds
    

class Text(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.text_enc = LanguageEmbeddingLayer()
        self.prj = SubNet(
            in_size=hp.d_tout,
            hidden_size=hp.d_tout,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
    
    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask, y=None):
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text_embedding = enc_word[:,0,:] # (batch_size, emb_size)
        text_out, preds = self.prj(text_embedding)
        return text_embedding, text_out, preds

class Acoustic(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.acoustic_enc = RNNEncoder(
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.prj = SubNet(
            in_size=hp.d_aout,
            hidden_size=hp.d_aout,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
    
    def forward(self, acoustic, a_len, y=None):
        acoustic_embedding = self.acoustic_enc(acoustic, a_len)
        acoustic_out, preds = self.prj(acoustic_embedding)
        return acoustic_embedding, acoustic_out, preds

class Visual(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.visual_enc = RNNEncoder(
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.prj = SubNet(
            in_size=hp.d_vout,
            hidden_size=hp.d_vout,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
    
    def forward(self, visual, v_len, y=None):
        visual_embedding = self.visual_enc(visual, v_len)
        visual_out, preds = self.prj(visual_embedding)
        return visual_embedding, visual_out, preds