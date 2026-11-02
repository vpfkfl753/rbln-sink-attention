"""Explicit opt-in kernel registration. Never runs on import."""
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import sys
from .contracts import UnsupportedContract

_LOADED=False

def register_kernels():
    global _LOADED
    if _LOADED:return
    if os.environ.get('RBLN_USE_CUSTOM_KERNEL')!='1':
        raise UnsupportedContract('Set RBLN_USE_CUSTOM_KERNEL=1 before starting the dedicated process')
    if os.environ.get('TRITON_CACHE_DIR'):
        raise UnsupportedContract('TRITON_CACHE_DIR override was not tested')
    try:version=importlib.metadata.version('rebel-compiler')
    except importlib.metadata.PackageNotFoundError as e:raise UnsupportedContract('Pinned RBLN SDK0.10.2 required; CPU reference tests do not need it') from e
    if version!='0.10.2':raise UnsupportedContract('Only SDK 0.10.2 was tested')
    import torch
    if torch.__version__!='2.9.1+cpu':raise UnsupportedContract('Tested device stack uses torch 2.9.1+cpu')
    names=[f'{family}_{phase}' for family in ('sliding_window_attention_naive','flash_causal_attention_naive') for phase in ('prefill','decode')]
    if any(hasattr(torch.ops.rbln_triton_ops,n) for n in names):
        raise UnsupportedContract('Op name collision: use a fresh dedicated process, never overwrite installed registrations')
    for name in ['swa_carrier_sink','full_once']:
        path=Path(__file__).parent/'vendor'/(name+'.py')
        spec=importlib.util.spec_from_file_location('rbln_sinks.vendor.'+name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    _LOADED=True
