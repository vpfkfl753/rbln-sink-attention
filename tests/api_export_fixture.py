"""CPU stub checks ABI/traceability only; no RBLN or NPU execution."""
import torch
from rbln_sinks import Contract,SinksModule
from test_cpu import inputs
lib=torch.library.Library('rbln_triton_ops','FRAGMENT')
for kind,prefix in [('swa','sliding_window_attention_naive'),('full','flash_causal_attention_naive')]:
 for phase in ['prefill','decode']:
  name=prefix+'_'+phase
  args='Tensor q, Tensor k, Tensor v, Tensor cache, Tensor seq, Tensor end, Tensor scale, Tensor bt, Tensor dummy' if kind=='swa' else 'Tensor q, Tensor k, Tensor v, Tensor cache, Tensor scale, Tensor seq, Tensor bt, Tensor dummy'
  lib.define(name+'('+args+') -> Tensor')
  def echo(q,*args):return q[...,:64].clone()
  lib.impl(name,echo,'CPU')
  def fake(q,*args):return torch.empty((*q.shape[:-1],64),dtype=q.dtype,device=q.device)
  torch.library.register_fake('rbln_triton_ops::'+name)(fake)
for kind in ['swa','full']:
 for phase in ['prefill','decode']:
  c=Contract(kind,phase,128 if phase=='prefill' else 1,torch.bfloat16,torch.bfloat16,128 if kind=='swa' else 1024)
  data=inputs(c);data['q'].normal_();c.validate_runtime(**data);module=SinksModule(c)
  out=module(**data);assert torch.equal(out,data['q'])
  ep=torch.export.export(module,(),kwargs=data)
  assert any('rbln_triton_ops' in str(n.target) for n in ep.graph.nodes)
print('Four CPU stub ABI/export checks passed; not device validation.')
