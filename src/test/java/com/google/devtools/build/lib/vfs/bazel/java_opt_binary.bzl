load("@with_cfg.bzl", "with_cfg")

java_opt_binary, _java_opt_binary = with_cfg(native.java_binary).set("compilation_mode", "opt").build()
