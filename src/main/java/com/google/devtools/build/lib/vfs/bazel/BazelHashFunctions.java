// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.vfs.bazel;

import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import java.security.Security;
import javax.annotation.Nullable;

/** Bazel specific {@link DigestHashFunction}s. */
public final class BazelHashFunctions {
  @Nullable public static final DigestHashFunction BLAKE3;

  static {
    DigestHashFunction hashFunction = null;

    if (JniLoader.isJniAvailable()) {
      try {
        Security.addProvider(new Blake3Provider());
        hashFunction = DigestHashFunction.register(new Blake3HashFunction(), "BLAKE3");
      } catch (UnsatisfiedLinkError ignored) {
        // This can happen if bazel was compiled manually (with compile.sh),
        // on windows. In that case jni is available, but missing the blake3
        // symbols necessary to register the hasher.
      }
    }

    BLAKE3 = hashFunction;
  }

  public static void ensureRegistered() {}

  private BazelHashFunctions() {}
}
