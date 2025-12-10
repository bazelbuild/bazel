"""Public go_binary macro that loads from private wrappers."""

load("go_wrappers.bzl", _go_binary_macro = "go_binary_macro")

go_binary = _go_binary_macro
