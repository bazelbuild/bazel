package com.google.devtools.build.lib.vfs;

import java.security.Provider;

public final class Blake3Provider extends Provider {
  public Blake3Provider() {
    super("BLAKE3Provider", "1.0", "A BLAKE3 digest provider");
    put("MessageDigest.BLAKE3", "com.google.devtools.build.lib.vfs.Blake3MessageDigest");
  }
}
