package com.google.devtools.build.lib.vfs.bazel;

import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import java.security.Security;

public final class BazelHashFunctions {
  public static final DigestHashFunction BLAKE3;

  static {
    DigestHashFunction hashFunction = null;

    if (JniLoader.isJniAvailable()) {
      try {
        Security.addProvider(new Blake3Provider());
        hashFunction = DigestHashFunction.register(new Blake3HashFunction(), "BLAKE3");
      } catch (java.lang.UnsatisfiedLinkError ignored) {
        // This can happen if bazel was compiled manually (with compile.sh),
        // on windows. In that case jni is available, but missing the blake3
        // symbols necessary to register the hasher.
      }
    }

    BLAKE3 = hashFunction;
  }

  public static void ensureRegistered() {}
  ;
}
