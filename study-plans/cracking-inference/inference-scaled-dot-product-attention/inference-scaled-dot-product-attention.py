import torch
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """

    # Transpose K, multiply QK
    trans_key = torch.transpose(key, 1, 2)

    raw_scores = torch.matmul(query, trans_key)
    
    # divide by sqrt(dk)
    dk = query.size(2)
    raw_scores = raw_scores * (1.0 / math.sqrt(dk))
    
    # Add M (???)
    if mask is not None:
        raw_scores = raw_scores.masked_fill(mask, float('-inf'))
    
    # softmax
    #softmax_result = torch.nn.Softmax(raw_scores, dim=0)
    func_softmax = torch.nn.Softmax(dim=-1)
    softmax_result = func_softmax(raw_scores)
    
    # multiply by V
    Y = torch.matmul(softmax_result, value)
    
    return Y
    #pass
