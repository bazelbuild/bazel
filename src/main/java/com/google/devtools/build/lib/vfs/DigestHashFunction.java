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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DigestLength.DigestLengthImpl;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map.Entry;

/**
 * Type of hash function to use for digesting files.
 *
 * <p>This tracks parallel {@link java.security.MessageDigest} and {@link HashFunction} interfaces
 * for each provided hash, as Bazel uses both - MessageDigest where performance is critical and
 * HashFunctions where ease-of-use wins over.
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

  private final HashFunction hashFunction;
  private final DigestLength digestLength;
  private final String name;
  private final MessageDigest messageDigestPrototype;
  private final boolean messageDigestPrototypeSupportsClone;
  private final ImmutableList<String> names;

  private DigestHashFunction(
      HashFunction hashFunction, DigestLength digestLength, ImmutableList<String> names) {
    this.hashFunction = hashFunction;
    this.digestLength = digestLength;
    checkArgument(!names.isEmpty());
    this.name = names.get(0);
    this.names = names;
    this.messageDigestPrototype = getMessageDigestInstance();
    this.messageDigestPrototypeSupportsClone = supportsClone(messageDigestPrototype);
  }

  public static DigestHashFunction register(
      HashFunction hash, String hashName, String... altNames) {
    return register(hash, new DigestLengthImpl(hash), hashName, altNames);
  }

  /**
   * Creates a new DigestHashFunction that is registered to be recognized by its name in {@link
   * DigestFunctionConverter}.
   *
   * @param hashName the canonical name for this hash function - and the name that can be used to
   *     uncover the MessageDigest.
   * @param altNames alternative names that will be mapped to this function by the converter but
   *     will not serve as the canonical name for the DigestHashFunction.
   * @param hash The {@link HashFunction} to register.
   * @throws IllegalArgumentException if the name is already registered.
   */
  public static DigestHashFunction register(
      HashFunction hash, DigestLength digestLength, String hashName, String... altNames) {
    try {
      MessageDigest.getInstance(hashName);
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalArgumentException(
          "The hash function name provided does not correspond to a valid MessageDigest: "
              + hashName,
          e);
    }

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

  /** Converts a string to its registered {@link DigestHashFunction}. */
  public static class DigestFunctionConverter extends Converter.Contextless<DigestHashFunction> {
    @Override
    public DigestHashFunction convert(String input) throws OptionsParsingException {
      for (Entry<String, DigestHashFunction> possibleFunctions : hashFunctionRegistry.entrySet()) {
        if (possibleFunctions.getKey().equalsIgnoreCase(input)) {
          return possibleFunctions.getValue();
        }
      }
      throw new OptionsParsingException(input + "is not a valid hash function.");
    }

    @Override
    public String getTypeDescription() {
      return "hash function";
    }
  }

  public HashFunction getHashFunction() {
    return hashFunction;
  }

  public MessageDigest cloneOrCreateMessageDigest() {
    if (messageDigestPrototypeSupportsClone) {
      try {
        return (MessageDigest) messageDigestPrototype.clone();
      } catch (CloneNotSupportedException e) {
        // We checked at initialization that this could be cloned, so this should never happen.
        throw new IllegalStateException("Could not clone message digest", e);
      }
    } else {
      return getMessageDigestInstance();
    }
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

  private MessageDigest getMessageDigestInstance() {
    try {
      return MessageDigest.getInstance(name);
    } catch (NoSuchAlgorithmException e) {
      // We check when we register() this digest function that the message digest exists. This
      // should never happen.
      throw new IllegalStateException("message digest " + name + " not available", e);
    }
  }

  private static boolean supportsClone(MessageDigest toCheck) {
    try {
      var unused = toCheck.clone();
      return true;
    } catch (CloneNotSupportedException e) {
      return false;
    }
  }

  public static ImmutableSet<DigestHashFunction> getPossibleHashFunctions() {
    return ImmutableSet.copyOf(hashFunctionRegistry.values());
  }
}
