
"""
SelfAttention
+------------------------------------------+
|                                          |
|     X    *     W_Q      =    Q           |
|     X    *     W_K      =    K           |
|     X    *     W_V      =    V           |
|                                          |
|  +-----+    +-----+       +-----+        |
|  |  dim     |  dim        |  dim         |
|  |seq       |dim          |seq           |
|  +          +             +              |
|                                          |
|     Q    *     K.t      =    score       |
|                                          |
|  +-----+    +-----+       +-----+        |
|  |  dim     |  seq        |  seq         |
|  |seq       |dim          |seq           |
|  +          +             +              |
|                                          |
|               score += mask (in Decoder) |
|                                          |
|  score = softmax(score/sqrt(d_k),axis=1) |
|                                          |
|   score  *      V       =    Z           |
|                                          |
|  +-----+    +-----+       +-----+        |
|  |  seq     |  dim        |  dim         |
|  |seq       |seq          |seq           |
|  +          +             +              |
|                                          |
+------------------------------------------+


Encoder Block
+----------------------------------------------+
|                                              |
| X_embed + Pos_encoding                       |
|         |                                    |
|         v                +-----------------+ |
|        X_embed +-------->|  SelfAttention  | |
|            +             +---+-------------+ |
|            |                 |               |
|            +---------+       |               |
|                      |       |               |
|                      |       |               |
|                  +---v-------v---+           |
|                  |  X_embed, Z1  |           |
|       LayerNorm( |  ResidualADD  | )         |
|            +     +---------------+           |
|            |                                 |
|            |                                 |
|           Z2---------+--------->             |
|                      | Linear( Z2 )          |
|                      |       |               |
|                  +---v-------v---+           |
|                  |   Z2,    Z3   |           |
| out = LayerNorm( |  ResidualADD  | )         |
|                  +---------------+           |
|                                              |
+----------------------------------------------+

X = Embed(X_in) + PosEnc
Z_sa = SelfAttention(Q, K, V)(X)
Z = LayerNorm(X + Z_sa)
Z_lin = LinearReLULinear(Z)
Z_out = LayerNorm(Z + Z_lin)

"""

