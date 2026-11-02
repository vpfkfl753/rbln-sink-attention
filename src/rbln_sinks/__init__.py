"""Sink-aware attention kernels for RBLN ATOM (research code, SDK 0.10.2)."""
from .contracts import Contract, UnsupportedContract
from .api import SinksModule, pack_query
from .reference import attention_reference, partition_reference
__all__=['Contract','UnsupportedContract','SinksModule','pack_query','attention_reference','partition_reference']
__version__='0.1.0'
