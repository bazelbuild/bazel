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
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Map;
import java.util.UUID;

import javax.annotation.Nullable;

/**
 * Simplified wrapper for MD5 message digests.
 *
 * @see java.security.MessageDigest
 */
public final class Fingerprint {

  private Hasher hasher;
  private static final HashFunction MD5_HASH_FUNCTION = Hashing.md5();

  /**
   * Creates and initializes a new Hasher.
   */
  public Fingerprint() {
    reset();
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
    byte[] bytes = hasher.hash().asBytes();
    reset();
    return bytes;
  }

  /**
   * Completes the hash computation and returns the digest as a string.
   *
   * <p>This method has the side-effect of resetting the underlying digest computer.
   *
   * @return the MD5 digest as a 32-character string of hexadecimal digits
   */
  public String hexDigestAndReset() {
    String hexDigest = hasher.hash().toString();
    reset();
    return hexDigest;
  }

  /**
   * Updates the digest with 0 or more bytes.
   *
   * @param input the array of bytes with which to update the digest
   * @see java.security.MessageDigest#update(byte[])
   */
  public Fingerprint addBytes(byte[] input) {
    hasher.putBytes(input);
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
    hasher.putBytes(input, offset, len);
    return this;
  }

  /**
   * Updates the digest with a boolean value.
   */
  public Fingerprint addBoolean(boolean input) {
    addBytes(new byte[] { (byte) (input ? 1 : 0) });
    return this;
  }

  /**
   * Updates the digest with a boolean value, correctly handling null.
   */
  public Fingerprint addNullableBoolean(Boolean input) {
    addInt(input == null ? -1 : (input.booleanValue() ? 1 : 0));
    return this;
  }

  /**
   * Updates the digest with the little-endian bytes of a given int value.
   *
   * @param input the integer with which to update the digest
   */
  public Fingerprint addInt(int input) {
    hasher.putInt(input);
    return this;
  }

  /**
   * Updates the digest with the little-endian bytes of a given long value.
   *
   * @param input the long with which to update the digest
   */
  public Fingerprint addLong(long input) {
    hasher.putLong(input);
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
    // Note that Hasher#putString() would not include the length of {@code input}.
    hasher.putBytes(bytes);
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
    hasher.putBytes(bytes);
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
    addStringLatin1(input.getPathString());
    return this;
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
   * Updates the digest with inputs by iterating over them and invoking
   * {@code #addString(String)} on each element.
   *
   * @param inputs the inputs with which to update the digest
   */
  public Fingerprint addStrings(String... inputs) {
    addInt(inputs.length);
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
    hasher = MD5_HASH_FUNCTION.newHasher();
  }

  // -------- Convenience methods ----------------------------

  /**
   * Computes the hex digest from a String using UTF8 encoding and returning
   * the hexDigest().
   *
   * @param input the String from which to compute the digest
   */
  public static String md5Digest(String input) {
    return MD5_HASH_FUNCTION.hashString(input, UTF_8).toString();
  }
}
