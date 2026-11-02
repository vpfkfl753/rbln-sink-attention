"""Normal Q + sinks API. Import has no device-registration side effects."""
import torch
from .contracts import Contract, require

def _pack(q, sinks):
    sf=sinks.to(torch.float32).reshape(1,8,8,1,1)
    pad=torch.full((1,8,8,1,63),float('-inf'),dtype=torch.float32,device=q.device)
    carrier=torch.cat([sf,pad],-1).expand(1,8,8,q.shape[-2],64)
    return torch.cat([q.to(torch.float32),carrier],-1)

def pack_query(q, sinks):
    """Host-only checked pack; D128 carrier, first D64 Q, then sink + 63 -inf."""
    require(q.ndim==5 and tuple(q.shape[:3])==(1,8,8) and q.shape[-1]==64 and q.shape[-2] in (1,64,128), 'Untested grouped Q')
    require(q.dtype in (torch.float16,torch.bfloat16,torch.float32), 'Unsupported query dtype')
    require(tuple(sinks.shape)==(64,) and sinks.dtype==torch.float32, 'Requires flat FP32 sinks')
    require(q.device.type==sinks.device.type=='cpu', 'Host CPU tensors required')
    require(q.is_contiguous() and sinks.is_contiguous(), 'Contiguous inputs required')
    require(bool(torch.isfinite(q).all()), 'Nonfinite Q')
    require(bool((torch.isfinite(sinks)|torch.isneginf(sinks)).all()), 'Invalid sink value')
    return _pack(q,sinks)

class SinksModule(torch.nn.Module):
    """Compiler-facing module; call contract.validate_runtime on real inputs first.

    The cache tensor must be marked static with CompileContext.mark_static_address.
    This module does NOT register kernels, construct runtimes, or manage KV state.
    """
    def __init__(self, contract: Contract):
        super().__init__()
        self.contract=contract
        self.register_buffer('scale',torch.tensor(0.125,dtype=torch.float32))

    def forward(self,q,k,v,cache,seq,bt,sinks,end=None):
        c=self.contract
        c.validate_static(q,k,v,cache,seq,bt,sinks,end)
        packed=_pack(q,sinks)
        if c.kind=='swa':
            op=getattr(torch.ops.rbln_triton_ops,'sliding_window_attention_naive_'+c.phase)
            return op(packed,k,v,cache,seq,end,self.scale,bt,self.scale).to(q.dtype)
        op=getattr(torch.ops.rbln_triton_ops,'flash_causal_attention_naive_'+c.phase)
        return op(packed,k,v,cache,self.scale,seq,bt,self.scale).to(q.dtype)
