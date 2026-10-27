"""Native adapters. Network access is confined to transports via ModelGateway."""
from .registry import ProviderRegistry, default_registry
from .types import ModelRequest, ModelRoute, ProviderState

__all__ = ["ProviderRegistry", "default_registry", "ModelRequest", "ModelRoute", "ProviderState"]
