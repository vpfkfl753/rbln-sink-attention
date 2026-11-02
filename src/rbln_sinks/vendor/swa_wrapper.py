"""Normal-Q + sinks wrapper for the sliding-window carrier kernel.

Only B1, D64, window 128 was tested on device. Other shapes are not routed.
"""
import torch

def pack(q,sinks):
    if q.ndim!=5 or q.shape[0]!=1 or q.shape[-1]!=64:
        raise ValueError('Expected B1 grouped Q with head dimension64')
    if sinks.numel()!=q.shape[1]*q.shape[2]:
        raise ValueError('One sink per query head required')
    # Keep sink precision through float32 packing, as in the original kernel.
    sf=sinks.to(torch.float32).reshape(1,q.shape[1],q.shape[2],1,1)
    pad=torch.full((1,q.shape[1],q.shape[2],1,63),float('-inf'),dtype=torch.float32,device=q.device)
    aligned=torch.cat([sf,pad],dim=-1).expand(1,q.shape[1],q.shape[2],q.shape[-2],64)
    return torch.cat([q.to(torch.float32),aligned],dim=-1)

def attention(q,k,v,cache,seq,end,scale,bt,sinks,decode=False):
    packed=pack(q,sinks)
    op=getattr(torch.ops.rbln_triton_ops,'sliding_window_attention_naive_'+('decode' if decode else 'prefill'))
    return op(packed,k,v,cache,seq,end,scale,bt,scale).to(q.dtype)
