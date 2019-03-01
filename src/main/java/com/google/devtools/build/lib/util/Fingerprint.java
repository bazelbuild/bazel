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

package com.google.devtools.build.lib.util;

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.DigestException;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Simplified wrapper for computing message digests.
 *
 * @see java.security.MessageDigest
 */
public final class Fingerprint implements Consumer<String> {

  // Make novel use of a CodedOutputStream, which is good at efficiently serializing data. By
  // flushing at the end of each digest we can continue to use the stream.
  private final CodedOutputStream codedOut;
  private final MessageDigest messageDigest;

  /** Creates and initializes a new instance. */
  public Fingerprint(DigestHashFunction digestFunction) {
    messageDigest = digestFunction.cloneOrCreateMessageDigest();
    // This is a lot of indirection, but CodedOutputStream does a reasonable job of converting
    // strings to bytes without creating a whole bunch of garbage, which pays off.
    codedOut =
        CodedOutputStream.newInstance(
            new DigestOutputStream(ByteStreams.nullOutputStream(), messageDigest),
            /*bufferSize=*/ 1024);
  }

  public Fingerprint() {
    // TODO(b/112460990): Use the value from DigestHashFunction.getDefault(), but check for
    // contention.
    this(DigestHashFunction.MD5);
  }

  /**
   * Completes the hash computation by doing final operations and resets the underlying state,
   * allowing this instance to be used again.
   *
   * @return the digest as a 16-byte array
   * @see java.security.MessageDigest#digest()
   */
  public byte[] digestAndReset() {
    try {
      codedOut.flush();
    } catch (IOException e) {
      throw new IllegalStateException("failed to flush", e);
    }
    return messageDigest.digest();
  }

  /**
   * Completes the hash computation by doing final operations and resets the underlying state,
   * allowing this instance to be used again.
   *
   * <p>Instead of returning a digest, this method writes the digest straight into the supplied byte
   * array, at the given offset.
   *
   * @see java.security.MessageDigest#digest()
   */
  public void digestAndReset(byte[] buf, int offset, int len) {
    try {
      codedOut.flush();
      messageDigest.digest(buf, offset, len);
    } catch (IOException e) {
      throw new IllegalStateException("failed to flush", e);
    } catch (DigestException e) {
      throw new IllegalStateException("failed to digest", e);
    }
  }

  /** Same as {@link #digestAndReset()}, except returns the digest in hex string form. */
  public String hexDigestAndReset() {
    return hexDigest(digestAndReset());
  }

  /** Updates the digest with 0 or more bytes. */
  public Fingerprint addBytes(byte[] input) {
    addBytes(input, 0, input.length);
    return this;
  }

  /** Updates the digest with the specified number of bytes starting at offset. */
  public Fingerprint addBytes(byte[] input, int offset, int len) {
    try {
      codedOut.write(input, offset, len);
    } catch (IOException e) {
      throw new IllegalStateException("failed to write bytes", e);
    }
    return this;
  }

  /** Updates the digest with a boolean value. */
  public Fingerprint addBoolean(boolean input) {
    try {
      codedOut.writeBoolNoTag(input);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    return this;
  }

  /** Same as {@link #addBoolean(boolean)}, except considers nullability. */
  public Fingerprint addNullableBoolean(Boolean input) {
    if (input == null) {
      addBoolean(false);
    } else {
      addBoolean(true);
      addBoolean(input);
    }
    return this;
  }

  /** Updates the digest with the varint representation of input. */
  public Fingerprint addInt(int input) {
    try {
      codedOut.writeInt32NoTag(input);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    return this;
  }

  /** Updates the digest with the signed varint representation of input. */
  Fingerprint addSInt(int input) {
    try {
      codedOut.writeSInt32NoTag(input);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    return this;
  }

  /** Updates the digest with the varint representation of a long value. */
  public Fingerprint addLong(long input) {
    try {
      codedOut.writeInt64NoTag(input);
    } catch (IOException e) {
      throw new IllegalStateException("failed to write long", e);
    }
    return this;
  }

  /** Same as {@link #addInt(int)}, except considers nullability. */
  public Fingerprint addNullableInt(@Nullable Integer input) {
    if (input == null) {
      addBoolean(false);
    } else {
      addBoolean(true);
      addInt(input);
    }
    return this;
  }

  /** Updates the digest with a UUID. */
  public Fingerprint addUUID(UUID uuid) {
    addLong(uuid.getLeastSignificantBits());
    addLong(uuid.getMostSignificantBits());
    return this;
  }

  /** Updates the digest with a String using UTF8 encoding. */
  public Fingerprint addString(String input) {
    try {
      codedOut.writeStringNoTag(input);
    } catch (IOException e) {
      throw new IllegalStateException("failed to write string", e);
    }
    return this;
  }

  /** Same as {@link #addString(String)}, except considers nullability. */
  public Fingerprint addNullableString(@Nullable String input) {
    if (input == null) {
      addBoolean(false);
    } else {
      addBoolean(true);
      addString(input);
    }
    return this;
  }

  /** Updates the digest with a {@link Path}. */
  public Fingerprint addPath(Path input) {
    addString(input.getPathString());
    return this;
  }

  /** Updates the digest with a {@link PathFragment}. */
  public Fingerprint addPath(PathFragment input) {
    return addString(input.getPathString());
  }

  /**
   * Add the supplied sequence of {@link String}s to the digest as an atomic unit, that is this is
   * different from adding them each individually.
   */
  public Fingerprint addStrings(Iterable<String> inputs) {
    int count = 0;
    for (String input : inputs) {
      addString(input);
      count++;
    }
    addInt(count);

    return this;
  }

  /**  Updates the digest with the supplied map. */
  public Fingerprint addStringMap(Map<String, String> inputs) {
    addInt(inputs.size());
    for (Map.Entry<String, String> entry : inputs.entrySet()) {
      addString(entry.getKey());
      addString(entry.getValue());
    }

    return this;
  }

  /**
   * Add the supplied sequence of {@link PathFragment}s to the digest as an atomic unit, that is
   * this is different from adding each item individually.
   *
   * @param inputs the paths with which to update the digest
   */
  public Fingerprint addPaths(Iterable<PathFragment> inputs) {
    int count = 0;
    for (PathFragment path : inputs) {
      addPath(path);
      count++;
    }
    addInt(count);

    return this;
  }

  private static String hexDigest(byte[] digest) {
    StringBuilder b = new StringBuilder(32);
    for (int i = 0; i < digest.length; i++) {
      int n = digest[i];
      b.append("0123456789abcdef".charAt((n >> 4) & 0xF));
      b.append("0123456789abcdef".charAt(n & 0xF));
    }
    return b.toString();
  }

  // -------- Convenience methods ----------------------------

  /**
   * Computes the hex digest from a String using UTF8 encoding and returning the hexDigest().
   *
   * @param input the String from which to compute the digest
   */
  public static String getHexDigest(String input) {
    // TODO(b/112460990): This convenience method, if kept should not use MD5 by default, but should
    // use the value from DigestHashFunction.getDefault(). However, this gets called during class
    // loading in a few places, before setDefault() has been called, so these call-sites should be
    // removed before this can be done safely.
    return hexDigest(
        DigestHashFunction.MD5
            .cloneOrCreateMessageDigest()
            .digest(input.getBytes(StandardCharsets.UTF_8)));
  }

  @Override
  public void accept(String s) {
    addString(s);
  }
}
