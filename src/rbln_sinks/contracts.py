"""Conservative structural whitelist; acceptance is not an FP64 accuracy pass."""
from dataclasses import dataclass
import torch

class UnsupportedContract(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise UnsupportedContract(message)

@dataclass(frozen=True)
class Contract:
    kind: str
    phase: str
    query: int
    dtype: torch.dtype
    cache_dtype: torch.dtype
    partition: int

    def __post_init__(self):
        require(self.kind in ('swa', 'full'), 'Only SWA128 and full causal attention')
        require(self.phase in ('prefill', 'decode'), 'Unknown phase')
        require(self.query == 1 if self.phase == 'decode' else self.query in (64,128), 'Untested query length')
        q = self.query
        if self.kind == 'swa':
            allowed = (self.partition == 128 and (
                (self.dtype == torch.float32 and self.cache_dtype == torch.float32 and q in (1,64,128)) or
                (self.dtype == torch.bfloat16 and self.cache_dtype == torch.float32 and q in (1,64)) or
                (self.dtype == torch.float16 and self.cache_dtype == torch.float32 and q in (1,64)) or
                (self.dtype == torch.bfloat16 and self.cache_dtype == torch.bfloat16 and q in (1,64,128))))
        else:
            allowed = ((self.partition == 128 and self.dtype == self.cache_dtype == torch.float32 and q in (1,64)) or
                       (self.partition == 1024 and self.dtype == self.cache_dtype == torch.bfloat16 and q in (1,128)))
        require(allowed, 'Untested dtype/cache/partition/query combination')

    @property
    def blocks(self):
        return 1 if self.kind == 'swa' else 2

    @property
    def table_dtype(self):
        return torch.int16 if self.kind == 'full' and self.partition == 1024 else torch.int32

    def validate_static(self, q, k, v, cache, seq, bt, sinks, end=None):
        """Trace-friendly metadata checks, no value reads; host value checks are separate."""
        require(tuple(q.shape) == (1,8,8,self.query,64), 'Requires B1/H64/KV8/G8/D64')
        for t in (k,v):
            require(tuple(t.shape) == (1,8,1,self.query,64), 'K/V grouped shape mismatch')
        require(tuple(cache.shape) == (2,self.blocks,8,1,self.partition,64), 'Static cache layout mismatch')
        require(all(t.dtype == self.dtype for t in (q,k,v)), 'QKV dtype mismatch')
        require(cache.dtype == self.cache_dtype, 'Cache dtype mismatch')
        require(tuple(sinks.shape) == (64,) and sinks.dtype == torch.float32, 'Requires flat FP32 sinks[64]')
        require(seq.dtype == torch.int16 and tuple(seq.shape) == ((1,) if self.kind == 'swa' else (1,2)), 'Sequence ABI mismatch')
        require(bt.dtype == self.table_dtype and tuple(bt.shape) == ((self.blocks,) if self.phase=='prefill' else (1,self.blocks)), 'Block-table ABI mismatch')
        ts = [q,k,v,cache,seq,bt,sinks]
        if self.kind == 'swa':
            require(end is not None and tuple(end.shape)==(1,) and end.dtype==torch.int16, 'SWA requires end[1] int16')
            ts.append(end)
        else:
            require(end is None, 'Full causal attention does not accept SWA end')
        require(all(t.device.type == 'cpu' for t in ts), 'Host inputs for this pinned RBLN compiler path must be CPU tensors')
        require(all(t.is_contiguous() for t in ts), 'Noncontiguous inputs are outside the tested ABI')

    def validate_runtime(self, q, k, v, cache, seq, bt, sinks, end=None):
        """Run on real host inputs BEFORE compilation/runtime invocation, never inside export.

        It cannot validate device cache contents or initialize them. The caller must
        retain the static cache identity and prime it through sequential calls.
        """
        self.validate_static(q,k,v,cache,seq,bt,sinks,end)
        require(all(bool(torch.isfinite(t).all()) for t in (q,k,v)), 'Nonfinite QKV unsupported')
        require(bool((torch.isfinite(sinks) | torch.isneginf(sinks)).all()), 'Sinks allow finite values or -inf only')
        require(bt.reshape(-1).tolist() == ([0] if self.kind=='swa' else [1,0]), 'Only observed physical block mapping supported')
        counts=seq.reshape(-1).tolist()
        require(all(0 <= n <= self.partition for n in counts), 'Invalid cached-prefix count')
        if self.kind=='full':
            require(counts[1]==0 or counts[0]==self.partition, 'Only dense causal prefix; no skipped leading partition')
            require(sum(counts)+self.query <= 2*self.partition, 'Cache capacity exceeded')
        else:
            require(int(end.item()) == counts[0]+self.query, 'SWA end must equal capped cached count + query')
        if self.phase=='prefill':
            require(sum(counts)%self.query == 0, 'Only observed aligned sequential prefill starts')
