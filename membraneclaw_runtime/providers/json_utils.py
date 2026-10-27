"""Strict shared JSON parsing; malformed bytes stay in raw trace payloads."""
import json
from ..tracing.schema import canonical_json
from .types import ProtocolError


def strict_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("non-finite JSON number")
    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
        canonical_json(value)
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ProtocolError("invalid provider JSON") from None
