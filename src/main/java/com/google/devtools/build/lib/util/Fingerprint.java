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
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * Simplified wrapper for MD5 message digests.
 *
 * @see java.security.MessageDigest
 */
public final class Fingerprint {

  private static final MessageDigest MD5_PROTOTYPE;
  private static final boolean MD5_PROTOTYPE_SUPPORTS_CLONE;

  static {
    MD5_PROTOTYPE = getMd5Instance();
    MD5_PROTOTYPE_SUPPORTS_CLONE = supportsClone(MD5_PROTOTYPE);
  }

  // Make novel use of a CodedOutputStream, which is good at efficiently serializing data. By
  // flushing at the end of each digest we can continue to use the stream.
  private final CodedOutputStream codedOut;
  private final MessageDigest md5;

  /** Creates and initializes a new instance. */
  public Fingerprint() {
    md5 = cloneOrCreateMd5();
    // This is a lot of indirection, but CodedOutputStream does a reasonable job of converting
    // strings to bytes without creating a whole bunch of garbage, which pays off.
    codedOut = CodedOutputStream.newInstance(
        new DigestOutputStream(ByteStreams.nullOutputStream(), md5),
        /*bufferSize=*/ 1024);
  }

  /**
   * Completes the hash computation by doing final operations and resets the underlying state,
   * allowing this instance to be used again.
   *
   * @return the MD5 digest as a 16-byte array
   * @see java.security.MessageDigest#digest()
   */
  public byte[] digestAndReset() {
    try {
      codedOut.flush();
    } catch (IOException e) {
      throw new IllegalStateException("failed to flush", e);
    }
    return md5.digest();
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
      throw new IllegalStateException();
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

  private static MessageDigest cloneOrCreateMd5() {
    if (MD5_PROTOTYPE_SUPPORTS_CLONE) {
      try {
        return (MessageDigest) MD5_PROTOTYPE.clone();
      } catch (CloneNotSupportedException e) {
        throw new IllegalStateException("Could not clone md5", e);
      }
    } else {
      return getMd5Instance();
    }
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

  private static MessageDigest getMd5Instance() {
    try {
      return MessageDigest.getInstance("md5");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("md5 not available", e);
    }
  }

  private static boolean supportsClone(MessageDigest toCheck) {
    try {
      toCheck.clone();
      return true;
    } catch (CloneNotSupportedException e) {
      return false;
    }
  }

  // -------- Convenience methods ----------------------------

  /**
   * Computes the hex digest from a String using UTF8 encoding and returning
   * the hexDigest().
   *
   * @param input the String from which to compute the digest
   */
  public static String md5Digest(String input) {
    return hexDigest(cloneOrCreateMd5().digest(input.getBytes(StandardCharsets.UTF_8)));
  }
}
