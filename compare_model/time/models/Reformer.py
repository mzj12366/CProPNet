import torch
import torch.nn as nn
import torch.nn.functional as F
from compare_model.time.layers.Transformer_EncDec import Encoder, EncoderLayer
from compare_model.time.layers.SelfAttention_Family import ReformerLayer
from compare_model.time.layers.Embed import DataEmbedding


class ReformerModel(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(ReformerModel, self).__init__()
        self.seq_len = configs.seq_len

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Classification head
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model * configs.seq_len, configs.model_output_dim)


    def classification(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, pe_info, ph_exam, x_enc, mask=None):

        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]

