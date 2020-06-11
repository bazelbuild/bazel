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

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.security.DigestException;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.util.Collection;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * Simplified wrapper for using {@link MessageDigest} to generate fingerprints.
 *
 * <p>A fingerprint is a cryptographic hash of a message that encodes the representation of an
 * object. Two objects of the same type have the same fingerprint if and only if they are equal.
 * This property allows fingerprints to be used as unique identifiers for objects of a particular
 * type, and for fingerprint equivalence to be used as a proxy for object equivalence, and these
 * properties hold even outside the process. Note that this is a stronger requirement than {@link
 * Object#hashCode}, which allows unequal objects to share the same hash code.
 *
 * <p>Values are added to the fingerprint by converting them to bytes and digesting the bytes.
 * Therefore, there are two potential sources of bugs: 1) a proper hash collision where two distinct
 * streams of bytes produce the same digest, and 2) a programming oversight whereby two unequal
 * values produce the same bytes, or conversely, two equal values produce distinct bytes.
 *
 * <p>The case of a hash collision is statistically very unlikely, so we just need to ensure a
 * one-to-one relationship between equality classes of values and their byte representation. A good
 * way to do this is to literally serialize the values such that there is enough information to
 * unambiguously deserialize them. For example, when serializing a list of strings ({@link
 * #addStrings}, it is enough to write each string's content along with its length, plus the overall
 * number of strings in the list. This ensures that no other list of strings can generate the same
 * bytes. Note that it is not necessary to avoid collisions between different fingerprinting methods
 * (e.g., between {@link #addStrings} and {@link #addString}) because the caller will only use one
 * or the other in a given context, or else the user is required to write a disambiguating tag if
 * both are possible.
 *
 * @see java.security.MessageDigest
 */
public final class Fingerprint {

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
    this(DigestHashFunction.SHA256);
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

  /**
   * Appends the specified bytes to the fingerprint message. Same as {@link #addBytes(byte[])}, but
   * faster when only a {@link ByteString} is available.
   *
   * <p>The fingerprint directly injects the bytes with no framing or tags added. Thus, not
   * guaranteed to be unambiguous; especially if input length is data-dependent.
   */
  public Fingerprint addBytes(ByteString bytes) {
    try {
      codedOut.writeRawBytes(bytes);
    } catch (IOException e) {
      throw new IllegalStateException("failed to write bytes", e);
    }
    return this;
  }

  /** Appends the specified bytes to the fingerprint message. */
  public Fingerprint addBytes(byte[] input) {
    addBytes(input, 0, input.length);
    return this;
  }

  /**
   * Appends the specified bytes to the fingerprint message, starting at offset.
   *
   * <p>The bytes are directly injected into the fingerprint with no framing or tags added. Thus,
   * not guaranteed to be unambiguous; especially if len is data-dependent.
   */
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
      throw new IllegalStateException("failed to write bool", e);
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

  /** Appends an int to the fingerprint message. */
  public Fingerprint addInt(int x) {
    try {
      codedOut.writeInt32NoTag(x);
    } catch (IOException e) {
      throw new IllegalStateException("failed to write int", e);
    }
    return this;
  }

  /** Appends a long to the fingerprint message. */
  public Fingerprint addLong(long x) {
    try {
      codedOut.writeInt64NoTag(x);
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

  /** Appends a {@link UUID} to the fingerprint message. */
  public Fingerprint addUUID(UUID uuid) {
    addLong(uuid.getLeastSignificantBits());
    addLong(uuid.getMostSignificantBits());
    return this;
  }

  /** Appends a String to the fingerprint message. */
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

  /** Appends a {@link Path} to the fingerprint message. */
  public Fingerprint addPath(Path input) {
    addString(input.getPathString());
    return this;
  }

  /** Appends a {@link PathFragment} to the fingerprint message. */
  public Fingerprint addPath(PathFragment input) {
    return addString(input.getPathString());
  }

  /**
   * Appends a collection of strings to the fingerprint message as a unit. The collection must have
   * a deterministic iteration order.
   *
   * <p>The fingerprint effectively records the sequence of calls, not just the elements. That is,
   * addStrings(x+y).addStrings(z) is different from addStrings(x).addStrings(y+z).
   */
  public Fingerprint addStrings(Collection<String> inputs) {
    addInt(inputs.size());
    for (String input : inputs) {
      addString(input);
    }

    return this;
  }

  /**
   * Appends an arbitrary sequence of Strings as a unit.
   *
   * <p>This is slightly less efficient than {@link #addStrings}.
   */
  // TODO(b/150312032): Deprecate this method.
  public Fingerprint addIterableStrings(Iterable<String> inputs) {
    for (String input : inputs) {
      addBoolean(true);
      addString(input);
    }
    addBoolean(false);

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

  /** Like {@link #addStrings} but for {@link PathFragment}. */
  public Fingerprint addPaths(Collection<PathFragment> inputs) {
    addInt(inputs.size());
    for (PathFragment input : inputs) {
      addPath(input);
    }
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
    // TODO(b/112460990): This convenience method should
    // use the value from DigestHashFunction.getDefault(). However, this gets called during class
    // loading in a few places, before setDefault() has been called, so these call-sites should be
    // removed before this can be done safely.
    return hexDigest(
        DigestHashFunction.SHA256.cloneOrCreateMessageDigest().digest(input.getBytes(UTF_8)));
  }
}
