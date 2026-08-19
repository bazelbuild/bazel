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
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache.KeyType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import org.apache.commons.lang3.StringUtils;

/** The content checksum for an HTTP download, which knows its own type. */
public class Checksum {

  public static final List<String> KNOWN_HASH_NAMES = Arrays.stream(KeyType.values())
      .map(KeyType::getHashName).toList();

  /** Exception thrown to indicate that a string is not a valid checksum for that key type. */
  public static final class InvalidChecksumException extends Exception {
    private InvalidChecksumException(KeyType keyType, String hash) {
      super("Invalid " + keyType + " checksum '" + hash + "'");
    }

    private InvalidChecksumException(String msg) {
      super(msg);
    }

    private InvalidChecksumException(String msg, Throwable cause) {
      super(msg, cause);
    }
  }

  /** Exception thrown to indicate that a checksum is missing. */
  public static final class MissingChecksumException extends IOException {
    public MissingChecksumException(String message) {
      super(message);
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

  private static byte[] base64Decode(String data) throws InvalidChecksumException {
    try {
      return Base64.getDecoder().decode(data);
    } catch (IllegalArgumentException e) {
      throw new InvalidChecksumException("Invalid base64 '" + data + "'", e);
    }
  }

  /**
   * Constructs a new Checksum from a hash in Subresource Integrity format.
   *
   * <p>Generally follows web's <a href="https://www.w3.org/TR/sri-2/">subresource integrity
   * spec</a> with the following differences:
   *
   * <ul>
   *   <li>Per the spec, multiple integrity hashes with the same hash algorithm are allowed. For
   *       Bazel, if there are multiple integrity hashes AND they are the strongest hash algorithm,
   *       an exception will be thrown. In Bazel, dependencies should be deterministic.
   *   <li>If there are no valid hash algorithms detected, web allows the resource to load. For
   *       Bazel, an error will be thrown since the intention of setting an integrity is to ensure
   *       that it is checked.
   * </ul>
   */
  public static Checksum fromSubresourceIntegrity(
      String integrity, @Nullable List<String> parseWarnings) throws InvalidChecksumException {
    if (!integrity.isEmpty() && integrity.isBlank()) {
      throw new InvalidChecksumException(String.format("Provided checksum is blank (%d whitespace characters)", integrity.length()));
    }

    Map<KeyType, List<Checksum>> metadata = parseMetadata(integrity, parseWarnings);
    if (metadata.isEmpty()) {
      throw new InvalidChecksumException(String.format("No valid checksums found in integrity '%s'", integrity));
    }

    @Nonnull KeyType strongestHashAlgo = findStrongestAlgorithm(metadata.keySet());
    List<Checksum> checksums = metadata.get(strongestHashAlgo);

    // The SRI spec allows multiple checksums with the "strongest" algorithm to allow for different
    // payloads from the same resource. This isn't allowed here to keep with the build philosophy of
    // reproducibility - a resource should always return the same content, so only a single checksum
    // per algorithm hash should be provided.
    if (checksums.size() > 1) {
      Checksum c1 = checksums.get(0);
      Checksum c2 = checksums.get(1);
      String sriFormat1 = String.format("%s-%s", c1.getKeyType().getHashName(), Base64.getEncoder()
          .encodeToString(c1.getHashCode().asBytes()));
      String sriFormat2 = String.format("%s-%s", c2.getKeyType().getHashName(), Base64.getEncoder()
          .encodeToString(c2.getHashCode().asBytes()));

      throw new InvalidChecksumException(
          String.format(
              "Duplicate hash algorithm in list of integrity hashes:\n\t%s\n\t%s", sriFormat1, sriFormat2));
    }

    return checksums.getFirst();
  }

  private static KeyType findStrongestAlgorithm(Set<KeyType> keyTypes) {
    Optional<KeyType> strongest =
        keyTypes.stream()
            .max(
                new Comparator<KeyType>() {
                  @Override
                  public int compare(KeyType o1, KeyType o2) {
                    return DownloadCache.DEFAULT_KEY_TYPE_ORDERING.indexOf(o1)
                        - DownloadCache.DEFAULT_KEY_TYPE_ORDERING.indexOf(o2);
                  }
                });
    return strongest.orElse(null);
  }

  public static Checksum fromSubresourceIntegrity(String integrity)
      throws InvalidChecksumException {
    return fromSubresourceIntegrity(integrity, null);
  }

  /**
   * Parses the subresource integrity field.
   *
   * <p>Generally follows the <a href="https://www.w3.org/TR/sri-2/#parse-metadata-section">SRI spec
   * for parsing</a>.
   *
   * @param integrityAttr The SRI integrity hash (multiple hashes allowed separated by whitespace).
   * @param parseWarnings If not null, errors in parsing are added to this list.
   */
  private static Map<KeyType, List<Checksum>> parseMetadata(
      String integrityAttr, @Nullable List<String> parseWarnings) {
    Map<KeyType, List<Checksum>> result = new HashMap<>();

    for (String integrityHash : StringUtils.split(integrityAttr)) {
      KeyType hashAlgorithm = parseHashAlgorithm(integrityHash);
      if (hashAlgorithm == null) {
        if (parseWarnings != null) {
          parseWarnings.add(
              String.format("Unknown hash algorithm for integrity: '%s'.", integrityHash));
        }
        continue;
      }

      // The SRI spec allows for future options on the hash after the '?' character. No actual
      // options are defined for now. Note them and ignore.
      String hashAndMaybeOptions = integrityHash.substring(hashAlgorithm.getHashName().length()+1);
      String hash = StringUtils.substringBefore(hashAndMaybeOptions, "?");
      String options = StringUtils.substringAfter(hashAndMaybeOptions, "?");

      if (!options.isEmpty() && parseWarnings != null) {
        parseWarnings.add(
            String.format(
                "Ignoring unknown integrity options '%s' from integrity '%s'.",
                options, integrityHash));
      }

      byte[] hashBytes = null;
      try {
        hashBytes = Base64.getDecoder().decode(hash);
      } catch (IllegalArgumentException e) {
        if (parseWarnings != null) {
          parseWarnings.add(String.format("Ignoring invalid base64 '%s'.", integrityHash));
        }
        continue;
      }

      Checksum checksum = null;
      try {
        checksum =
            Checksum.fromString(
                hashAlgorithm,
                HashCode.fromBytes(hashBytes).toString(),
                /* useSubresourceIntegrity= */ true);
      } catch (InvalidChecksumException e) {
        if (parseWarnings != null) {
          parseWarnings.add(String.format("Ignoring invalid checksum '%s'.", integrityHash));
        }
        continue;
      }

      if (!result.containsKey(hashAlgorithm)) {
        result.put(hashAlgorithm, new ArrayList<>());
      }
      result.get(hashAlgorithm).add(checksum);
    }
    return result;
  }

  /**
   * Parses the {@link KeyType} from the given integrity hash.
   *
   * <p>Eg. Given the integrity hash:
   *
   * <pre>
   * <code>sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=</code> returns {@link KeyType.KeyType.SHA256}
   * <code>invalid-prefixNotValidUnknown</code> returns {@code null}
   * <code>sha256</code> returns {@code null}
   * </pre>
   */
  private static KeyType parseHashAlgorithm(String integrity) {
    for (String hashName : KNOWN_HASH_NAMES) {
      if (integrity.startsWith(hashName)
          && integrity.length() > hashName.length()
          && integrity.charAt(hashName.length()) == '-') {
        return KeyType.getByHashName(hashName);
      }
    }

    return null;
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

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }
    if (other instanceof Checksum c) {
      return keyType.equals(c.keyType) && hashCode.equals(c.hashCode);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return hashCode.hashCode() * 31 + keyType.hashCode();
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
