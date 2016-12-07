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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
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

  private static final byte[] TRUE_BYTES = new byte[] { 1 };
  private static final byte[] FALSE_BYTES = new byte[] { 0 };

  private static final MessageDigest MD5_PROTOTYPE;
  private static final boolean MD5_PROTOTYPE_SUPPORTS_CLONE;

  static {
    MD5_PROTOTYPE = getMd5Instance();
    MD5_PROTOTYPE_SUPPORTS_CLONE = supportsClone(MD5_PROTOTYPE);
  }

  private final ByteBuffer scratch;
  private final MessageDigest md5;

  /**
   * Creates and initializes a new instance.
   */
  public Fingerprint() {
    scratch = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    md5 = cloneOrCreateMd5();
  }

  /**
   * Completes the hash computation by doing final operations, e.g., padding.
   *
   * <p>This method has the side-effect of resetting the underlying digest computer.
   *
   * @return the MD5 digest as a 16-byte array
   * @see java.security.MessageDigest#digest()
   */
  public byte[] digestAndReset() {
    // Reset is implicit.
    return md5.digest();
  }

  /**
   * Completes the hash computation and returns the digest as a string.
   *
   * <p>This method has the side-effect of resetting the underlying digest computer.
   *
   * @return the MD5 digest as a 32-character string of hexadecimal digits
   */
  public String hexDigestAndReset() {
    return hexDigest(digestAndReset());
  }

  /**
   * Updates the digest with 0 or more bytes.
   *
   * @param input the array of bytes with which to update the digest
   * @see java.security.MessageDigest#update(byte[])
   */
  public Fingerprint addBytes(byte[] input) {
    md5.update(input);
    return this;
  }

  /**
   * Updates the digest with the specified number of bytes starting at offset.
   *
   * @param input the array of bytes with which to update the digest
   * @param offset the offset into the array
   * @param len the number of bytes to use
   * @see java.security.MessageDigest#update(byte[], int, int)
   */
  public Fingerprint addBytes(byte[] input, int offset, int len) {
    md5.update(input, offset, len);
    return this;
  }

  /**
   * Updates the digest with a boolean value.
   */
  public Fingerprint addBoolean(boolean input) {
    md5.update(input ? TRUE_BYTES : FALSE_BYTES);
    return this;
  }

  /**
   * Updates the digest with a boolean value, correctly handling null.
   */
  public Fingerprint addNullableBoolean(Boolean input) {
    return addInt(input == null ? -1 : (input.booleanValue() ? 1 : 0));
  }

  /**
   * Updates the digest with the little-endian bytes of a given int value.
   *
   * @param input the integer with which to update the digest
   */
  public Fingerprint addInt(int input) {
    scratch.putInt(input);
    updateFromScratch(4);
    return this;
  }

  /**
   * Updates the digest with the little-endian bytes of a given long value.
   *
   * @param input the long with which to update the digest
   */
  public Fingerprint addLong(long input) {
    scratch.putLong(input);
    updateFromScratch(8);
    return this;
  }

  /**
   * Updates the digest with the little-endian bytes of a given int value, correctly distinguishing
   * between null and non-null values.
   *
   * @param input the integer with which to update the digest
   */
  public Fingerprint addNullableInt(@Nullable Integer input) {
    if (input == null) {
      addInt(0);
    } else {
      addInt(1);
      addInt(input);
    }
    return this;
  }

  /**
   * Updates the digest with a UUID.
   *
   * @param uuid the UUID with which to update the digest. Must not be null.
   */
  public Fingerprint addUUID(UUID uuid) {
    addLong(uuid.getLeastSignificantBits());
    addLong(uuid.getMostSignificantBits());
    return this;
  }

  /**
   * Updates the digest with a String using its length plus its UTF8 encoded bytes.
   *
   * @param input the String with which to update the digest
   * @see java.security.MessageDigest#update(byte[])
   */
  public Fingerprint addString(String input) {
    byte[] bytes = input.getBytes(UTF_8);
    addInt(bytes.length);
    md5.update(bytes);
    return this;
  }

  /**
   * Updates the digest with a String using its length plus its UTF8 encoded bytes; if the string
   * is null, then it uses -1 as the length.
   *
   * @param input the String with which to update the digest
   * @see java.security.MessageDigest#update(byte[])
   */
  public Fingerprint addNullableString(@Nullable String input) {
    if (input == null) {
      addInt(-1);
    } else {
      addString(input);
    }
    return this;
  }

  /**
   * Updates the digest with a String using its length and content.
   *
   * @param input the String with which to update the digest
   * @see java.security.MessageDigest#update(byte[])
   */
  public Fingerprint addStringLatin1(String input) {
    addInt(input.length());
    byte[] bytes = new byte[input.length()];
    for (int i = 0; i < input.length(); i++) {
      bytes[i] = (byte) input.charAt(i);
    }
    md5.update(bytes);
    return this;
  }

  /**
   * Updates the digest with a Path.
   *
   * @param input the Path with which to update the digest.
   */
  public Fingerprint addPath(Path input) {
    addStringLatin1(input.getPathString());
    return this;
  }

  /**
   * Updates the digest with a Path.
   *
   * @param input the Path with which to update the digest.
   */
  public Fingerprint addPath(PathFragment input) {
    return addStringLatin1(input.getPathString());
  }

  /**
   * Updates the digest with inputs by iterating over them and invoking
   * {@code #addString(String)} on each element.
   *
   * @param inputs the inputs with which to update the digest
   */
  public Fingerprint addStrings(Iterable<String> inputs) {
    addInt(Iterables.size(inputs));
    for (String input : inputs) {
      addString(input);
    }

    return this;
  }

  /**
   * Updates the digest with inputs which are pairs in a map, by iterating over
   * the map entries and invoking {@code #addString(String)} on each key and
   * value.
   *
   * @param inputs the inputs in a map with which to update the digest
   */
  public Fingerprint addStringMap(Map<String, String> inputs) {
    addInt(inputs.size());
    for (Map.Entry<String, String> entry : inputs.entrySet()) {
      addString(entry.getKey());
      addString(entry.getValue());
    }

    return this;
  }

  /**
   * Updates the digest with a list of paths by iterating over them and
   * invoking {@link #addPath(PathFragment)} on each element.
   *
   * @param inputs the paths with which to update the digest
   */
  public Fingerprint addPaths(Iterable<PathFragment> inputs) {
    addInt(Iterables.size(inputs));
    for (PathFragment path : inputs) {
      addPath(path);
    }

    return this;
  }

  /**
   * Reset the Fingerprint for additional use as though previous digesting had not been done.
   */
  public void reset() {
    md5.reset();
  }

  private void updateFromScratch(int numBytes) {
    md5.update(scratch.array(), 0, numBytes);
    scratch.clear();
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
