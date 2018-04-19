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
package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.base.Preconditions;
import com.google.common.collect.MapMaker;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationConstants;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Supports association between fingerprints and NestedSet contents. A single NestedSetStore
 * instance should be globally available across a single process.
 *
 * <p>Maintains the fingerprint -> contents side of the bimap by decomposing nested Object[]'s.
 *
 * <p>For example, suppose the NestedSet A can be drawn as:
 *
 * <pre>
 *         A
 *       /  \
 *      B   C
 *     / \
 *    D  E
 * </pre>
 *
 * <p>Then, in memory, A = [[D, E], C]. To store the NestedSet, we would rely on the fingerprint
 * value FPb = fingerprint([D, E]) and write
 *
 * <pre>A -> fingerprint(FPb, C)</pre>
 *
 * <p>On retrieval, A will be reconstructed by first retrieving A using its fingerprint, and then
 * recursively retrieving B using its fingerprint.
 */
public class NestedSetStore {

  /** Stores fingerprint -> NestedSet associations. */
  interface NestedSetStorageEndpoint {
    /** Associates a fingerprint with the serialized representation of some NestedSet contents. */
    void put(ByteString fingerprint, byte[] serializedBytes);

    /**
     * Retrieves the serialized bytes for the NestedSet contents associated with this fingerprint.
     */
    byte[] get(ByteString fingerprint);
  }

  /** An in-memory {@link NestedSetStorageEndpoint} */
  private static class InMemoryNestedSetStorageEndpoint implements NestedSetStorageEndpoint {
    private final ConcurrentHashMap<ByteString, byte[]> fingerprintToContents =
        new ConcurrentHashMap<>();

    @Override
    public void put(ByteString fingerprint, byte[] serializedBytes) {
      fingerprintToContents.put(fingerprint, serializedBytes);
    }

    @Override
    public byte[] get(ByteString fingerprint) {
      return fingerprintToContents.get(fingerprint);
    }
  }

  /** An in-memory cache for fingerprint <-> NestedSet associations. */
  private static class NestedSetCache {
    private final Map<ByteString, Object> fingerprintToContents =
        new MapMaker()
            .concurrencyLevel(SerializationConstants.DESERIALIZATION_POOL_SIZE)
            .weakValues()
            .makeMap();

    /** Object/Object[] contents to fingerprint. Maintained for fast fingerprinting. */
    private final Map<Object, ByteString> contentsToFingerprint =
        new MapMaker()
            .concurrencyLevel(SerializationConstants.DESERIALIZATION_POOL_SIZE)
            .weakKeys()
            .makeMap();

    /**
     * Returns the NestedSet contents associated with the given fingerprint. Returns null if the
     * fingerprint is not known.
     */
    @Nullable
    public Object contentsForFingerprint(ByteString fingerprint) {
      return fingerprintToContents.get(fingerprint);
    }

    /**
     * Retrieves the fingerprint associated with the given NestedSet contents, or null if the given
     * contents are not known.
     */
    @Nullable
    public ByteString fingerprintForContents(Object contents) {
      return contentsToFingerprint.get(contents);
    }

    /** Associates the provided fingerprint and NestedSet contents. */
    public void put(ByteString fingerprint, Object contents) {
      contentsToFingerprint.put(contents, fingerprint);
      fingerprintToContents.put(fingerprint, contents);
    }
  }

  private final NestedSetCache nestedSetCache = new NestedSetCache();
  private final NestedSetStorageEndpoint nestedSetStorageEndpoint =
      new InMemoryNestedSetStorageEndpoint();

  /**
   * Computes and returns the fingerprint for the given NestedSet contents using the given {@link
   * SerializationContext}, while also associating the contents with the computed fingerprint in the
   * store. Recursively does the same for all transitive members (i.e. Object[] members) of the
   * provided contents.
   */
  ByteString computeFingerprintAndStore(Object contents, SerializationContext serializationContext)
      throws SerializationException {
    ByteString priorFingerprint = nestedSetCache.fingerprintForContents(contents);
    if (priorFingerprint != null) {
      return priorFingerprint;
    }

    // For every fingerprint computation, we need to use a new memoization table.  This is required
    // to guarantee that the same child will always have the same fingerprint - otherwise,
    // differences in memoization context could cause part of a child to be memoized in one
    // fingerprinting but not in the other.  We expect this clearing of memoization state to be a
    // major source of extra work over the naive serialization approach.  The same value may have to
    // be serialized many times across separate fingerprintings.
    SerializationContext newSerializationContext = serializationContext.getNewMemoizingContext();
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    CodedOutputStream codedOutputStream = CodedOutputStream.newInstance(byteArrayOutputStream);

    try {
      if (contents instanceof Object[]) {
        Object[] contentsArray = (Object[]) contents;
        codedOutputStream.writeInt32NoTag(contentsArray.length);
        for (Object child : contentsArray) {
          if (child instanceof Object[]) {
            ByteString fingerprint = computeFingerprintAndStore(child, serializationContext);
            newSerializationContext.serialize(fingerprint, codedOutputStream);
          } else {
            newSerializationContext.serialize(child, codedOutputStream);
          }
        }
      } else {
        codedOutputStream.writeInt32NoTag(1);
        newSerializationContext.serialize(contents, codedOutputStream);
      }
      codedOutputStream.flush();
    } catch (IOException e) {
      throw new SerializationException("Could not serialize " + contents, e);
    }

    byte[] serializedBytes = byteArrayOutputStream.toByteArray();
    ByteString fingerprint =
        ByteString.copyFrom(Hashing.md5().hashBytes(serializedBytes).asBytes());

    nestedSetCache.put(fingerprint, contents);
    nestedSetStorageEndpoint.put(fingerprint, serializedBytes);

    return fingerprint;
  }

  /** Retrieves and deserializes the NestedSet contents associated with the given fingerprint. */
  public Object getContentsAndDeserialize(
      ByteString fingerprint, DeserializationContext deserializationContext)
      throws IOException, SerializationException {
    Object contents = nestedSetCache.contentsForFingerprint(fingerprint);
    if (contents != null) {
      return contents;
    }

    byte[] retrieved =
        Preconditions.checkNotNull(
            nestedSetStorageEndpoint.get(fingerprint),
            "Fingerprint %s not found in NestedSetStore",
            fingerprint);
    CodedInputStream codedIn = CodedInputStream.newInstance(retrieved);
    DeserializationContext newDeserializationContext =
        deserializationContext.getNewMemoizingContext();

    int numberOfElements = codedIn.readInt32();
    Object dereferencedContents;
    if (numberOfElements > 1) {
      Object[] dereferencedContentsArray = new Object[numberOfElements];
      for (int i = 0; i < numberOfElements; i++) {
        Object deserializedElement = newDeserializationContext.deserialize(codedIn);
        dereferencedContentsArray[i] =
            deserializedElement instanceof ByteString
                ? getContentsAndDeserialize(
                    (ByteString) deserializedElement, deserializationContext)
                : deserializedElement;
      }
      dereferencedContents = dereferencedContentsArray;
    } else {
      dereferencedContents = newDeserializationContext.deserialize(codedIn);
    }

    nestedSetCache.put(fingerprint, dereferencedContents);
    return dereferencedContents;
  }
}
