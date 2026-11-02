"""FP64 mathematical oracle over logical history, not a device cache emulator."""
import math
import torch

def attention_reference(q,k_history,v_history,sinks,*,start,window=None,scale=0.125):
    # q[B,H,Q,D], history[B,KV,T,D]; GQA contiguous head groups.
    if q.ndim!=4 or k_history.ndim!=4 or v_history.shape!=k_history.shape:
        raise ValueError('Expected Q[B,H,Q,D] and matching K/V[B,KV,T,D]')
    b,h,n,d=q.shape;bk,g,t,dk=k_history.shape
    if bk!=b or dk!=d or h%g or not 0<=start or start+n>t or window is not None and window<=0:
        raise ValueError('Invalid GQA or logical-history bounds')
    if sinks.numel()!=h:raise ValueError('One denominator sink per query head')
    if not bool((torch.isfinite(sinks)|torch.isneginf(sinks)).all()):raise ValueError('Invalid sinks')
    # Convert AFTER input quantization by caller; do not regenerate unrounded inputs.
    q=q.double();k=k_history.double().repeat_interleave(h//g,dim=1);v=v_history.double().repeat_interleave(h//g,dim=1)
    qp=torch.arange(start,start+n,device=q.device)[:,None];kp=torch.arange(t,device=q.device)[None,:]
    mask=kp<=qp
    if window is not None:mask=mask & (kp>qp-window)
    z=(q@k.transpose(-1,-2)*scale).masked_fill(~mask,float('-inf'))
    zs=sinks.double().reshape(1,h,1,1).expand(b,h,n,1)
    probabilities=torch.cat([z,zs],-1).softmax(-1)
    return probabilities[...,:t]@v  # sink has no value vector

def partition_reference(q,k,v,sinks,*,start,partition,scale=0.125,repeat_sink=False):
    """Independent stable online softmax fold; repeat_sink is mathematical negative control."""
    b,h,n,d=q.shape;g=k.shape[1];t=k.shape[2]
    if t%partition or h%g:raise ValueError('Require whole partitions and grouped heads')
    q=q.double();k=k.double().repeat_interleave(h//g,1);v=v.double().repeat_interleave(h//g,1)
    maximum=torch.full((b,h,n,1),-math.inf,dtype=torch.float64)
    denominator=torch.zeros_like(maximum);numerator=torch.zeros((b,h,n,d),dtype=torch.float64)
    for lo in range(0,t,partition):
        hi=lo+partition;z=q@k[:,:,lo:hi].transpose(-1,-2)*scale
        mask=torch.arange(lo,hi)[None,:] <= torch.arange(start,start+n)[:,None]
        z=z.masked_fill(~mask,-math.inf)
        sink=sinks.double().reshape(1,h,1,1).expand(b,h,n,1) if lo==0 or repeat_sink else torch.full_like(maximum,-math.inf)
        new_max=torch.maximum(maximum,torch.maximum(z.max(-1,keepdim=True).values,sink))
        factor=torch.exp(maximum-new_max);weights=torch.exp(z-new_max)
        numerator=numerator*factor+weights@v[:,:,lo:hi]
        denominator=denominator*factor+weights.sum(-1,keepdim=True)+torch.exp(sink-new_max)
        maximum=new_max
    return numerator/denominator
