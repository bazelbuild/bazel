// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.vfs;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DigestLength.DigestLengthImpl;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map.Entry;

/**
 * Type of hash function to use for digesting files.
 */
// The underlying HashFunctions are immutable and thread safe.
public class DigestHashFunction {
  // This map must be declared first to make sure that calls to register() have it ready.
  private static final HashMap<String, DigestHashFunction> hashFunctionRegistry = new HashMap<>();

  /** Describes the length of a digest. */
  public interface DigestLength {
    /** Returns the length of a digest by inspecting its bytes. Used for variable-length digests. */
    default int getDigestLength(byte[] bytes, int offset) {
      return getDigestMaximumLength();
    }

    /** Returns the maximum length a digest can turn into. */
    int getDigestMaximumLength();

    /** Default implementation that simply returns a fixed length. */
    class DigestLengthImpl implements DigestLength {
      private final int length;

      DigestLengthImpl(HashFunction hashFunction) {
        this.length = hashFunction.bits() / 8;
      }

      @Override
      public int getDigestMaximumLength() {
        return length;
      }
    }
  }

  public static final DigestHashFunction SHA1 = register(Hashing.sha1(), "SHA-1", "SHA1");
  public static final DigestHashFunction SHA256 = register(Hashing.sha256(), "SHA-256", "SHA256");
  public static final DigestHashFunction SHA384 = register(Hashing.sha384(), "SHA-384", "SHA384");
  public static final DigestHashFunction SHA512 = register(Hashing.sha512(), "SHA-512", "SHA512");

  private static DigestHashFunction defaultHash;

  private final HashFunction hashFunction;
  private final DigestLength digestLength;
  private final String name;
  private final ImmutableList<String> names;

  private DigestHashFunction(
      HashFunction hashFunction, DigestLength digestLength, ImmutableList<String> names) {
    this.hashFunction = hashFunction;
    this.digestLength = digestLength;
    checkArgument(!names.isEmpty());
    this.name = names.get(0);
    this.names = names;
  }

  public static DigestHashFunction register(
      HashFunction hash, String hashName, String... altNames) {
    return register(hash, new DigestLengthImpl(hash), hashName, altNames);
  }

  /**
   * Creates a new DigestHashFunction that is registered to be recognized by its name in {@link
   * DigestFunctionConverter}.
   *
   * @param hashName the canonical name for this hash function.
   * @param altNames alternative names that will be mapped to this function by the converter but
   *     will not serve as the canonical name for the DigestHashFunction.
   * @param hash The {@link HashFunction} to register.
   * @throws IllegalArgumentException if the name is already registered.
   */
  public static DigestHashFunction register(
      HashFunction hash, DigestLength digestLength, String hashName, String... altNames) {
    ImmutableList<String> names =
        ImmutableList.<String>builder().add(hashName).add(altNames).build();
    DigestHashFunction hashFunction = new DigestHashFunction(hash, digestLength, names);
    synchronized (hashFunctionRegistry) {
      for (String name : names) {
        if (hashFunctionRegistry.containsKey(name)) {
          throw new IllegalArgumentException("Hash function " + name + " is already registered.");
        }
        hashFunctionRegistry.put(name, hashFunction);
      }
    }
    return hashFunction;
  }

  /**
   * Returns the default DigestHashFunction for this instance of Bazel.
   *
   * <p>Note: This is a synchronized function, to make sure it does not occur concurrently with
   * {@link #setDefault(DigestHashFunction)}. Once this value is set, it's a constant, so to prevent
   * blocking calls, users should cache this value if needed.
   *
   * @throws DefaultHashFunctionNotSetException if the default has not yet been set by a previous
   *     call to {@link #setDefault}.
   */
  public static synchronized DigestHashFunction getDefault()
      throws DefaultHashFunctionNotSetException {
    DigestHashFunction hash = defaultHash;
    if (hash == null) {
      throw new DefaultHashFunctionNotSetException("DigestHashFunction default has not been set");
    }
    return hash;
  }

  /**
   * Returns the default DigestHashFunction, or the testing default if unset.
   */
  public static DigestHashFunction getDefaultUnchecked() {
    try {
      return getDefault();
    } catch (DefaultHashFunctionNotSetException e) {
      // Some tests use this class without calling GoogleUnixFileSystemModule.globalInit().
      Preconditions.checkState(
          System.getenv("TEST_TMPDIR") != null, "Default hash function has not been set");
      return DigestHashFunction.SHA256;
    }
  }


  /** Indicates that the default has not been initialized. */
  public static final class DefaultHashFunctionNotSetException extends Exception {
    DefaultHashFunctionNotSetException(String message) {
      super(message);
    }
  }

  /**
   * Sets the default DigestHashFunction for this instance of Bazel - can only be set once to
   * prevent incongruities.
   *
   * @throws DefaultAlreadySetException if it was already set.
   */
  public static synchronized void setDefault(DigestHashFunction hash)
      throws DefaultAlreadySetException {
    Preconditions.checkNotNull(hash);
    // Permit redundant calls.  This is difficult to avoid with test suites.
    if (defaultHash == null || defaultHash == hash) {
      defaultHash = hash;
      return;
    }
    throw new DefaultAlreadySetException(
        String.format(
            "setDefault(%s) failed. The default has already been set to %s, you cannot change it.",
            hash.name, defaultHash.name));
  }

  /** Failure to set the default if the default already being set. */
  public static final class DefaultAlreadySetException extends Exception {
    DefaultAlreadySetException(String message) {
      super(message);
    }
  }

  /** Converts a string to its registered {@link DigestHashFunction}. */
  public static class DigestFunctionConverter implements Converter<DigestHashFunction> {
    @Override
    public DigestHashFunction convert(String input) throws OptionsParsingException {
      for (Entry<String, DigestHashFunction> possibleFunctions : hashFunctionRegistry.entrySet()) {
        if (possibleFunctions.getKey().equalsIgnoreCase(input)) {
          return possibleFunctions.getValue();
        }
      }
      throw new OptionsParsingException("Not a valid hash function.");
    }

    @Override
    public String getTypeDescription() {
      return "hash function";
    }
  }

  public HashFunction getHashFunction() {
    return hashFunction;
  }

  public DigestLength getDigestLength() {
    return digestLength;
  }

  public ImmutableList<String> getNames() {
    return names;
  }

  @Override
  public String toString() {
    return name;
  }

  @VisibleForTesting
  static Collection<DigestHashFunction> getPossibleHashFunctions() {
    return hashFunctionRegistry.values();
  }
}
