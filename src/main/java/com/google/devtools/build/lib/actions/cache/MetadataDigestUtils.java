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
   * An empty digest of the same length as a real metadata digest, used as the seed (and additive
   * identity for {@link DigestUtils#combineUnordered}) of {@link #fromMetadata}.
   */
  private static final byte[] EMPTY_DIGEST = new Fingerprint().digestAndReset();

  /**
   * @param source the byte buffer source.
   * @return the digest from the given buffer.
   */
  public static byte[] read(ByteBuffer source) throws IOException {
    int size = VarInt.getVarInt(source);
    if (size < 0) {
      throw new IOException("Negative digest size: " + size);
    }
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
   * Computes an order-independent digest from the given (path, metadata) pairs.
   *
   * @param mdMap A collection of (execPath, FileArtifactValue) pairs. Values may be null.
   */
  public static byte[] fromMetadata(Map<String, FileArtifactValue> mdMap) {
    return fromMetadata(mdMap, EMPTY_DIGEST);
  }

  /**
   * Computes an order-independent digest from the given (path, metadata) pairs, combined into the
   * given seed digest.
   *
   * <p>Because {@link DigestUtils#combineUnordered} is commutative and associative, folding one
   * disjoint subset of entries into {@link #EMPTY_DIGEST} and then folding the remaining entries
   * into that result yields the same digest as folding all entries at once. The action cache
   * exploits this to avoid re-hashing an action's mandatory inputs: it seeds with the precomputed
   * mandatory-inputs digest (itself a {@code fromMetadata} over just those inputs) and folds in only
   * the discovered inputs and outputs.
   *
   * @param mdMap A collection of (execPath, FileArtifactValue) pairs. Values may be null.
   * @param seed the digest to combine the entries into; not mutated.
   */
  public static byte[] fromMetadata(Map<String, FileArtifactValue> mdMap, byte[] seed) {
    // combineUnordered may clobber its longer argument, so start from a copy of the seed to avoid
    // mutating a digest that the caller may retain (e.g., a stored mandatory-inputs digest).
    byte[] result = seed.clone();
    // Profiling showed that MessageDigest engine instantiation was a hotspot, so create one
    // instance for this computation to amortize its cost.
    Fingerprint fp = new Fingerprint();
    for (Map.Entry<String, FileArtifactValue> entry : mdMap.entrySet()) {
      result =
          DigestUtils.combineUnordered(result, getDigest(fp, entry.getKey(), entry.getValue()));
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
