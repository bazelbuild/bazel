// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.VarInt;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Map;

/**
 * A value class for capturing and comparing MD5-based digests.
 *
 * <p>Note that this class is responsible for digesting file metadata in an
 * order-independent manner. Care must be taken to do this properly. The
 * digest must be a function of the set of (path, metadata) tuples. While the
 * order of these pairs must not matter, it would <b>not</b> be safe to make
 * the digest be a function of the set of paths and the set of metadata.
 *
 * <p>Note that the (path, metadata) tuples must be unique, otherwise the
 * XOR-based approach will fail.
 */
public class Digest {

  static final int MD5_SIZE = 16;

  private final byte[] digest;

  /**
   * Construct the digest from the given bytes.
   * @param digest an MD5 digest. Must be sized properly.
   */
  private Digest(byte[] digest) {
    this.digest = digest;
  }

  /**
   * @param source the byte buffer source.
   * @return the digest from the given buffer.
   * @throws IOException if the byte buffer is incorrectly formatted.
   */
  public static Digest read(ByteBuffer source) throws IOException {
    int size = VarInt.getVarInt(source);
    if (size != MD5_SIZE) {
      throw new IOException("Unexpected digest length: " + size);
    }
    byte[] bytes = new byte[size];
    source.get(bytes);
    return new Digest(bytes);
  }

  /**
   * Write the digest to the output stream.
   */
  public void write(OutputStream sink) throws IOException {
    VarInt.putVarInt(digest.length, sink);
    sink.write(digest);
  }

  /**
   * @param mdMap A collection of (execPath, Metadata) pairs.
   *              Values may be null.
   * @return an <b>order-independent</b> digest from the given "set" of
   *         (path, metadata) pairs.
   */
  public static Digest fromMetadata(Map<String, Metadata> mdMap) {
    byte[] result = new byte[MD5_SIZE];
    // Profiling showed that MD5 engine instantiation was a hotspot, so create one instance for
    // this computation to amortize its cost.
    Fingerprint fp = new Fingerprint();
    for (Map.Entry<String, Metadata> entry : mdMap.entrySet()) {
      xorWith(result, getDigest(fp, entry.getKey(), entry.getValue()));
    }
    return new Digest(result);
  }

  /**
   * @return this Digest as a Metadata with no mtime.
   */
  public Metadata asMetadata() {
    return new Metadata(digest);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(digest);
  }

  @Override
  public boolean equals(Object obj) {
    return (obj instanceof Digest) && Arrays.equals(digest, ((Digest) obj).digest);
  }

  @Override
  public String toString() {
    return HashCode.fromBytes(digest).toString();
  }

  private static byte[] getDigest(Fingerprint fp, String execPath, Metadata md) {
    fp.addStringLatin1(execPath);

    if (md == null) {
      // Move along, nothing to see here.
    } else if (md.digest == null) {
      // Use the timestamp if the digest is not present, but not both.
      // Modifying a timestamp while keeping the contents of a file the
      // same should not cause rebuilds.
      fp.addLong(md.mtime);
    } else {
      fp.addBytes(md.digest);
    }
    return fp.digestAndReset();
  }

  /**
   * Compute lhs ^= rhs bitwise operation of the arrays.
   */
  private static void xorWith(byte[] lhs, byte[] rhs) {
    for (int i = 0; i < lhs.length; i++) {
      lhs[i] ^= rhs[i];
    }
  }
}
