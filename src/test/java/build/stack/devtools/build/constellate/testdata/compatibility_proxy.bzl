"""Compatibility proxy that wraps native rules.

This simulates a common pattern where rules are re-exported through
a compatibility layer.
"""

# Re-export native rules through this proxy
java_binary = native.java_binary
java_library = native.java_library
java_test = native.java_test
