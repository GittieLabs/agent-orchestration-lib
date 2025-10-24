"""Event adapters for different notification channels.

This module contains adapters for routing events to external systems like
logging frameworks, metrics systems, and webhooks.
"""

from .structured_logging import StructuredLogAdapter
from .metrics import MetricsAdapter
from .webhook import WebhookAdapter

__all__ = [
    "StructuredLogAdapter",
    "MetricsAdapter",
    "WebhookAdapter",
]
