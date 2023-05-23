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

package com.google.devtools.build.lib.hash;

import com.google.devtools.build.lib.jni.JniLoader;
import java.nio.ByteBuffer;

public class Blake3JNI {
  private Blake3JNI() {}

  static {
    JniLoader.loadJni();
  }

  public static final native long allocate_hasher();

  public static final native void delete_hasher(long self);

  public static final native void blake3_hasher_init(long self);

  public static final native void blake3_hasher_init_keyed(long self, byte[] key);

  public static final native void blake3_hasher_init_derive_key(long self, String context);

  public static final native void blake3_hasher_update(long self, ByteBuffer inputBuf, int offset, int input_len);

  public static final native void blake3_hasher_finalize(long self, byte[] out, int out_len);

  public static final native void blake3_hasher_finalize_seek(
      long self, long seek, byte[] out, int out_len);
}
