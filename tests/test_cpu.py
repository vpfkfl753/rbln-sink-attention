import importlib.util,json,math,unittest,subprocess,sys
from pathlib import Path
import torch
from rbln_sinks import Contract, UnsupportedContract, pack_query, attention_reference, partition_reference, SinksModule
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

def inputs(c):
 q=torch.zeros(1,8,8,c.query,64,dtype=c.dtype);k=torch.zeros(1,8,1,c.query,64,dtype=c.dtype)
 cache=torch.zeros(2,c.blocks,8,1,c.partition,64,dtype=c.cache_dtype)
 seq=torch.zeros((1,) if c.kind=='swa' else (1,2),dtype=torch.int16)
 bt=torch.tensor([0] if c.kind=='swa' else [1,0],dtype=c.table_dtype)
 if c.phase=='decode':bt=bt.reshape(1,-1)
 return dict(q=q,k=k,v=k.clone(),cache=cache,seq=seq,bt=bt,sinks=torch.zeros(64),end=torch.tensor([c.query],dtype=torch.int16) if c.kind=='swa' else None)

class CPUContracts(unittest.TestCase):
 def test_pack_exact_historical_wrapper(self):
  spec=importlib.util.spec_from_file_location('archived',ROOT/'src/rbln_sinks/vendor/swa_wrapper.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  for dtype in [torch.bfloat16,torch.float16,torch.float32]:
   for length in [1,64,128]:
    with self.subTest(dtype=dtype,length=length):
     q=torch.randn(1,8,8,length,64,dtype=dtype);s=torch.linspace(-5,6,64);q0=q.clone();s0=s.clone();y=pack_query(q,s)
     self.assertTrue(torch.equal(y,old.pack(q,s)))
     self.assertEqual(y.dtype,torch.float32);self.assertTrue(y.is_contiguous())
     self.assertTrue(torch.equal(y[...,:64],q.float()));self.assertTrue(torch.isneginf(y[...,65:]).all())
     self.assertTrue(torch.equal(y[...,64],s.reshape(1,8,8,1).expand(1,8,8,length)))
     self.assertTrue(torch.equal(q,q0));self.assertTrue(torch.equal(s,s0))
 def test_disabled_pack(self):
  y=pack_query(torch.zeros(1,8,8,1,64),torch.full((64,),-math.inf));self.assertTrue(torch.isneginf(y[...,64:]).all())
 def test_invalid_pack(self):
  q=torch.zeros(1,8,8,1,64);s=torch.zeros(64)
  variants=[(q.expand(2,8,8,1,64),s),(q[...,:32],s),(q,s[:32]),(q,s.double()),(q,torch.full((64,),math.nan)),(q,torch.full((64,),math.inf)),(q.double(),s)]
  for x,y in variants:
   with self.subTest(shape=x.shape,dtype=x.dtype):
    with self.assertRaises(UnsupportedContract):pack_query(x,y)
 def test_profile_whitelist(self):
  for kind,dtype,cache,p,qs in [('swa',torch.bfloat16,torch.bfloat16,128,[1,64,128]),('swa',torch.float16,torch.float32,128,[1,64]),('swa',torch.bfloat16,torch.float32,128,[1,64]),('swa',torch.float32,torch.float32,128,[1,64,128]),('full',torch.float32,torch.float32,128,[1,64]),('full',torch.bfloat16,torch.bfloat16,1024,[1,128])]:
   for q in qs:
    c=Contract(kind,'decode' if q==1 else 'prefill',q,dtype,cache,p);c.validate_runtime(**inputs(c))
  for args in [('swa','prefill',32,torch.float32,torch.float32,128),('full','prefill',128,torch.float32,torch.float32,1024),('full','decode',1,torch.bfloat16,torch.bfloat16,128),('swa','decode',1,torch.float32,torch.float16,128),('swa','prefill',64,torch.float32,torch.float32,256)]:
   with self.assertRaises(UnsupportedContract):Contract(*args)
 def test_runtime_rejections(self):
  c=Contract('swa','decode',1,torch.bfloat16,torch.bfloat16,128)
  for field,change in [('q',lambda x:x[:,:4].contiguous()),('seq',lambda x:torch.tensor([-1],dtype=torch.int16)),('bt',lambda x:x+1),('end',lambda x:x+1),('k',lambda x:x.float()),('sinks',lambda x:torch.full_like(x,math.inf)),('q',lambda x:torch.full_like(x,math.nan)),('cache',lambda x:x.float())]:
   d=inputs(c);d[field]=change(d[field])
   with self.subTest(field=field):
    with self.assertRaises(UnsupportedContract):c.validate_runtime(**d)
 def test_dense_prefix_and_capacity(self):
  c=Contract('full','decode',1,torch.bfloat16,torch.bfloat16,1024);d=inputs(c)
  for counts in [[0,1],[1024,1024],[1025,0]]:
   d['seq']=torch.tensor([counts],dtype=torch.int16)
   with self.assertRaises(UnsupportedContract):c.validate_runtime(**d)
  d['seq']=torch.tensor([[1024,1]],dtype=torch.int16);c.validate_runtime(**d)
 def test_compiler_api_cpu_stub(self):
  subprocess.run([sys.executable,str(ROOT/"tests/api_export_fixture.py")],check=True,capture_output=True,text=True)
 def test_no_registration_on_import(self):
  self.assertFalse(hasattr(torch.ops.rbln_triton_ops,'sliding_window_attention_naive_prefill'))
 def test_device_registration_denied_without_sdk(self):
  from rbln_sinks.device import register_kernels
  with self.assertRaises(UnsupportedContract):register_kernels()

class Mathematics(unittest.TestCase):
 def test_sink_denominator_only(self):
  q=torch.zeros(1,2,1,1);k=torch.zeros(1,1,1,1);v=torch.tensor([[[[6.]]]])
  out=attention_reference(q,k,v,torch.zeros(2),start=0,scale=1)
  self.assertTrue(torch.equal(out,torch.full_like(out,3)))
  off=attention_reference(q,k,v,torch.full((2,),-math.inf),start=0,scale=1)
  self.assertTrue(torch.equal(off,torch.full_like(off,6)))
 def test_self_inclusive_causal_and_window(self):
  q=torch.zeros(1,1,1,1);k=torch.zeros(1,1,4,1);v=torch.tensor([[[[2.],[4.],[8.],[999.]]]])
  off=torch.tensor([-math.inf]);swa=attention_reference(q,k,v,off,start=2,window=2)
  self.assertAlmostEqual(swa.item(),6);full=attention_reference(q,k,v,off,start=2);self.assertAlmostEqual(full.item(),14/3)
 def test_gqa_mapping(self):
  q=torch.zeros(1,4,1,1);k=torch.zeros(1,2,1,1);v=torch.tensor([[[[2.]],[[8.]]]])
  self.assertEqual(attention_reference(q,k,v,torch.full((4,),-math.inf),start=0).reshape(-1).tolist(),[2.,2.,8.,8.])
 def test_partition_once_and_wrong_repeated(self):
  gen=torch.Generator().manual_seed(1249)
  for partition in [2,4,8]:
   for start,n in [(0,1),(7,1),(4,4)]:
    q=torch.randn(1,4,n,4,generator=gen);k=torch.randn(1,2,8,4,generator=gen);v=torch.randn(1,2,8,4,generator=gen);s=torch.tensor([4.,5.,6.,4.])
    a=attention_reference(q,k,v,s,start=start);b=partition_reference(q,k,v,s,start=start,partition=partition)
    self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))
    if partition<8:self.assertFalse(torch.allclose(a,partition_reference(q,k,v,s,start=start,partition=partition,repeat_sink=True),atol=1e-8))
 def test_stable_high_score(self):
  q=torch.tensor([[[[1000.]]]]);k=torch.tensor([[[[1000.]]]]);v=torch.ones_like(k)
  y=attention_reference(q,k,v,torch.tensor([1000000.]),start=0,scale=1);self.assertEqual(y.item(),.5)
 def test_real_quantized_fixture_reference(self):
  f=torch.load(ROOT/'fixtures/real_op_fixture.pt',weights_only=True);s=64
  q=f['q'][:,:,s:s+1].bfloat16();k=f['k'][:,:,:s+1].bfloat16();v=f['v'][:,:,:s+1].bfloat16()
  y=attention_reference(q,k,v,torch.full((64,),-math.inf),start=s,window=128).reshape(1,8,8,1,64)
  archived=torch.load(ROOT/'fixtures/swa_decode_disabled64.pt',weights_only=True)
  self.assertTrue(torch.allclose(y,archived['reference64'],atol=1e-12,rtol=1e-12))
  # BF16 disabled-sink decode at position 64 is within the 0.05 gate; the learned-sink case is not.
  self.assertLess((archived['actual'].double()-y).abs().max().item(),.05)
  learned=attention_reference(q,k,v,f['sinks'].reshape(-1).float(),start=64,window=128).reshape(1,8,8,1,64)
  failed=torch.load(ROOT/'fixtures/swa_decode_learned64.pt',weights_only=True)
  self.assertTrue(torch.allclose(learned,failed['reference64'],atol=1e-12,rtol=1e-12))
  self.assertGreaterEqual((failed['actual'].double()-learned).abs().max().item(),.05)

if __name__=='__main__':unittest.main(verbosity=2)
