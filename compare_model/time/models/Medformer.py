import torch
import torch.nn as nn
import torch.nn.functional as F
from compare_model.time.layers.Medformer_EncDec import Encoder, EncoderLayer
from compare_model.time.layers.SelfAttention_Family import MedformerLayer
from compare_model.time.layers.Embed import ListPatchEmbedding


class MedformerModel(nn.Module):
    def __init__(self, configs):
        super(MedformerModel, self).__init__()
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel
        
        # Embedding
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            configs.single_channel,
        )
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        
        # Classification head
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model
            * sum(patch_num_list)
            * (1 if not self.single_channel else configs.enc_in),
            configs.model_output_dim,
        )

    def forward(self,  info, exam,  x_enc, padding_mask=None):
        """
        Forward pass for classification task
        
        Args:
            x_enc: Input tensor of shape (batch_size, seq_len, enc_in)
            padding_mask: Padding mask (not used in current implementation)
            
        Returns:
            output: Classification logits of shape (batch_size, num_classes)
        """
        # Embedding
        enc_inputs = self.enc_embedding(x_enc)  # list of [B, patch_num_i, d_model]
        # Pass list into encoder (expected by Medformer encoder)
        enc_out, attns = self.encoder(enc_inputs, attn_mask=None)
        # Concatenate encoder outputs across patch streams for classification head
        if isinstance(enc_out, list):
            enc_out = torch.cat(enc_out, dim=1)  # [B, sum(patch_num_i), d_model]
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
