// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.base.Ascii;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import java.util.Base64;

/** The content checksum for an HTTP download, which knows its own type. */
public class Checksum {
  /** Exception thrown to indicate that a string is not a valid checksum for that key type. */
  public static final class InvalidChecksumException extends Exception {
    private InvalidChecksumException(KeyType keyType, String hash) {
      super("Invalid " + keyType + " checksum '" + hash + "'");
    }

    private InvalidChecksumException(String msg) {
      super(msg);
    }
  }

  private final KeyType keyType;
  private final HashCode hashCode;
  private final boolean useSubresourceIntegrity;

  private Checksum(KeyType keyType, HashCode hashCode, boolean useSubresourceIntegrity) {
    this.keyType = keyType;
    this.hashCode = hashCode;
    this.useSubresourceIntegrity = useSubresourceIntegrity;
  }

  /** Constructs a new Checksum for a given key type and hash, in hex format. */
  public static Checksum fromString(KeyType keyType, String hash) throws InvalidChecksumException {
    return fromString(keyType, hash, /* useSubresourceIntegrity= */ false);
  }

  private static Checksum fromString(KeyType keyType, String hash, boolean useSubresourceIntegrity)
      throws InvalidChecksumException {
    if (!keyType.isValid(hash)) {
      throw new InvalidChecksumException(keyType, hash);
    }
    return new Checksum(
        keyType, HashCode.fromString(Ascii.toLowerCase(hash)), useSubresourceIntegrity);
  }

  /** Constructs a new Checksum from a hash in Subresource Integrity format. */
  public static Checksum fromSubresourceIntegrity(String integrity)
      throws InvalidChecksumException {
    Base64.Decoder decoder = Base64.getDecoder();
    KeyType keyType = null;
    byte[] hash = null;
    int expectedLength = 0;

    if (integrity.startsWith("sha1-")) {
      keyType = KeyType.SHA1;
      expectedLength = 20;
      hash = decoder.decode(integrity.substring(5));
    }
    if (integrity.startsWith("sha256-")) {
      keyType = KeyType.SHA256;
      expectedLength = 32;
      hash = decoder.decode(integrity.substring(7));
    }
    if (integrity.startsWith("sha384-")) {
      keyType = KeyType.SHA384;
      expectedLength = 48;
      hash = decoder.decode(integrity.substring(7));
    }
    if (integrity.startsWith("sha512-")) {
      keyType = KeyType.SHA512;
      expectedLength = 64;
      hash = decoder.decode(integrity.substring(7));
    }
    if (integrity.startsWith("blake3-")) {
      keyType = KeyType.BLAKE3;
      expectedLength = 32;
      hash = decoder.decode(integrity.substring(7));
    }

    if (keyType == null) {
      throw new InvalidChecksumException(
          "Unsupported checksum algorithm: '"
              + integrity
              + "' (expected SHA-1, SHA-256, SHA-384, or SHA-512)");
    }

    if (hash.length != expectedLength) {
      throw new InvalidChecksumException(
          "Invalid " + keyType + " SRI checksum '" + integrity + "'");
    }

    return Checksum.fromString(
        keyType, HashCode.fromBytes(hash).toString(), /* useSubresourceIntegrity= */ true);
  }

  private static String toSubresourceIntegrity(KeyType keyType, HashCode hashCode) {
    String encoded = Base64.getEncoder().encodeToString(hashCode.asBytes());
    return keyType.getHashName() + "-" + encoded;
  }

  public String toSubresourceIntegrity() {
    return toSubresourceIntegrity(keyType, hashCode);
  }

  @Override
  public String toString() {
    return hashCode.toString();
  }

  public HashCode getHashCode() {
    return hashCode;
  }

  public KeyType getKeyType() {
    return keyType;
  }

  public String emitOtherHashInSameFormat(HashCode otherHash) {
    if (useSubresourceIntegrity) {
      return toSubresourceIntegrity(keyType, otherHash);
    } else {
      return otherHash.toString();
    }
  }
}
