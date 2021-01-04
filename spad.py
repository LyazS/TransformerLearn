from TransformerModule import Transformer
import torch
from tr_utils import SeqBatch
src = torch.randint(0, 20, (48, 20))
tgt = torch.randint(0, 20, (48, 20))
sb = SeqBatch(src, tgt)
print(sb.src_mask.shape, sb.tgt_mask.shape)
model = Transformer(23, 23, 6, 512, 256, 8)
# q = torch.randn(10, 33, 512)
# k = torch.randn(10, 33, 512)

# z = model(q, k, None, None)
z = model(sb.src, sb.tgt, sb.src_mask, sb.tgt_mask)
print(z.shape)