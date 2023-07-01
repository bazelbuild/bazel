package com.google.devtools.build.lib.vfs.bazel;

import java.security.Provider;

public final class Blake3Provider extends Provider {
  public Blake3Provider() {
    super("BLAKE3Provider", "1.0", "A BLAKE3 digest provider");
    put("MessageDigest.BLAKE3", Blake3MessageDigest.class.getName());
  }
}
