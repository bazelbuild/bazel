// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.VarInt;
import com.google.devtools.build.lib.vfs.DigestUtils;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Map;
import javax.annotation.Nullable;

/** Utility class for digests/metadata relating to the action cache. */
public final class MetadataDigestUtils {
  private MetadataDigestUtils() {}

  /**
   * @param source the byte buffer source.
   * @return the digest from the given buffer.
   */
  public static byte[] read(ByteBuffer source) {
    int size = VarInt.getVarInt(source);
    byte[] bytes = new byte[size];
    source.get(bytes);
    return bytes;
  }

  /** Write the digest to the output stream. */
  public static void write(byte[] digest, OutputStream sink) throws IOException {
    VarInt.putVarInt(digest.length, sink);
    sink.write(digest);
  }

  /**
   * @param mdMap A collection of (execPath, FileArtifactValue) pairs. Values may be null.
   * @return an <b>order-independent</b> digest from the given "set" of (path, metadata) pairs.
   */
  public static byte[] fromMetadata(Map<String, FileArtifactValue> mdMap) {
    byte[] result = new byte[1]; // reserve the empty string
    // Profiling showed that MessageDigest engine instantiation was a hotspot, so create one
    // instance for this computation to amortize its cost.
    Fingerprint fp = new Fingerprint();
    for (Map.Entry<String, FileArtifactValue> entry : mdMap.entrySet()) {
      result = DigestUtils.xor(result, getDigest(fp, entry.getKey(), entry.getValue()));
    }
    return result;
  }

  /**
   * @param env A collection of (String, String) pairs.
   * @return an order-independent digest of the given set of pairs.
   */
  public static byte[] fromEnv(Map<String, String> env) {
    byte[] result = new byte[0];
    Fingerprint fp = new Fingerprint();
    for (Map.Entry<String, String> entry : env.entrySet()) {
      fp.addString(entry.getKey());
      fp.addString(entry.getValue());
      result = DigestUtils.xor(result, fp.digestAndReset());
    }
    return result;
  }

  private static byte[] getDigest(Fingerprint fp, String execPath, @Nullable FileArtifactValue md) {
    fp.addString(execPath);
    if (md != null) {
      md.addTo(fp);
    }
    return fp.digestAndReset();
  }
}
