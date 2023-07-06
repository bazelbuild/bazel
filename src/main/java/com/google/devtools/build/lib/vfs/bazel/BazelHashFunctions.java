package com.google.devtools.build.lib.vfs.bazel;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import java.security.Security;

public final class BazelHashFunctions {
  static {
    Security.addProvider(new Blake3Provider());
  }

  public static final DigestHashFunction BLAKE3 =
      DigestHashFunction.register(new Blake3HashFunction(), "BLAKE3");

  public static void ensureRegistered() {}
  ;
}
