from Model.Utils import (
    InputEmbeddings,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForwardBlock,
    ProjectionLayer
)
from Model.Encoder import (
    Encoder,
    EncoderBlock
)
from Model.Decoder import (
    Decoder,
    DecoderBlock
)
from .transformer import Transformer

import torch.nn as nn

def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048
):
    src_embed = InputEmbeddings(
        d_model,
        src_vocab_size
    )
    tgt_embed = InputEmbeddings(
        d_model,
        tgt_vocab_size
    )

    src_pos = PositionalEncoding(
        d_model,
        src_seq_len,
        dropout
    )
    tgt_pos = PositionalEncoding(
        d_model,
        tgt_seq_len,
        dropout
    )

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(
            d_model=d_model,
            h=h,
            dropout=dropout
        )
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block,
            feed_forward_block,
            dropout
        )
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(
        nn.ModuleList(
            encoder_blocks
        )
    )
    decoder = Decoder(
        nn.ModuleList(
            decoder_blocks
        )
    )

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer