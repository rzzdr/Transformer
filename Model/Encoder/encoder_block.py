import torch.nn as nn
from Model.Utils import (
    MultiHeadAttention,
    FeedForwardBlock,
    ResidualConnection
)

class EncoderBlock(nn.Module):
    def __init__(
            self,
            self_attention_block: MultiHeadAttention,
            feed_forward_block: FeedForwardBlock,
            dropout: float
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [
                ResidualConnection(
                    dropout
                ) for _ in range(2)
            ]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x,
            lambda x: self.self_attention_block(
                x,
                x,
                x,
                src_mask
            )
        )
        x = self.residual_connection[1](
            x,
            self.feed_forward_block
        )

        return x