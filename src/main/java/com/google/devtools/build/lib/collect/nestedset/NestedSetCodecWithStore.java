// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.hash.Hashing;
import com.google.common.hash.HashingOutputStream;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationConstants;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Codec for {@link NestedSet} that uses the {@link NestedSetStore}.
 *
 * <p>Currently not used in favor of an @{@link AutoCodec}-ed NestedSet. Disabled by just not ending
 * in "Codec".
 */
public class NestedSetCodecWithStore<T> implements ObjectCodec<NestedSet<T>> {

  private final ConcurrentHashMap<ByteString, Object> fingerprintToContents =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<Object, ByteString> contentsToFingerprint =
      new ConcurrentHashMap<>();

  @SuppressWarnings("unchecked")
  @Override
  public Class<NestedSet<T>> getEncodedClass() {
    // Compiler doesn't like cast from Class<NestedSet> -> Class<NestedSet<T>>, but it
    // does allow what we see below. Type is lost at runtime anyway, so while gross this works.
    return (Class<NestedSet<T>>) ((Class<?>) NestedSet.class);
  }

  @Override
  public void serialize(SerializationContext context, NestedSet<T> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!SerializationConstants.shouldSerializeNestedSet) {
      // Don't perform NestedSet serialization in testing
      return;
    }

    context.serialize(obj.getOrder(), codedOut);
    // If the NestedSet is empty, it needs to be assigned to the EMPTY_CHILDREN constant on
    // deserialization.
    codedOut.writeBoolNoTag(obj.isEmpty());
    if (obj.isEmpty()) {
      return;
    }
    ByteString fingerprint = serializeToFingerprint(obj.rawChildren(), context);
    codedOut.writeByteArrayNoTag(fingerprint.toByteArray());
  }

  @Override
  public NestedSet<T> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    if (!SerializationConstants.shouldSerializeNestedSet) {
      // Don't perform NestedSet deserialization in testing
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    Order order = context.deserialize(codedIn);
    boolean isEmpty = codedIn.readBool();
    if (isEmpty) {
      return NestedSetBuilder.emptySet(order);
    }

    ByteString fingerprint = ByteString.copyFrom(codedIn.readByteArray());
    Object members = fingerprintToContents.get(fingerprint);
    return new NestedSet<>(order, members);
  }

  private ByteString serializeToFingerprint(
      Object children, SerializationContext serializationContext) throws SerializationException {
    // For every fingerprint computation, we need to use a new memoization table.  This is required
    // to guarantee that the same child will always have the same fingerprint - otherwise,
    // differences in memoization context could cause part of a child to be memoized in one
    // fingerprinting but not in the other.  We expect this clearing of memoization state to be a
    // major source of extra work over the naive serialization approach.  The same value may have to
    // be serialized many times across separate fingerprintings.
    SerializationContext newSerializationContext = serializationContext.getNewMemoizingContext();

    HashingOutputStream hashingOutputStream =
        new HashingOutputStream(Hashing.md5(), ByteStreams.nullOutputStream());
    CodedOutputStream codedOutputStream = CodedOutputStream.newInstance(hashingOutputStream);

    try {
      if (children instanceof Object[]) {
        for (Object child : (Object[]) children) {
          if (child instanceof Object[]) {
            ByteString fingerprint = contentsToFingerprint.get(child);
            // If this fingerprint is not yet known, we recurse to compute it.
            if (fingerprint == null) {
              fingerprint = serializeToFingerprint(child, serializationContext);
            }
            codedOutputStream.writeBytesNoTag(fingerprint);
          } else {
            newSerializationContext.serialize(child, codedOutputStream);
          }
        }
      } else {
        newSerializationContext.serialize(children, codedOutputStream);
      }
      codedOutputStream.flush();
    } catch (IOException e) {
      throw new SerializationException(
          "Could not serialize " + children + ": " + e.getMessage(), e);
    }

    ByteString fingerprint = ByteString.copyFrom(hashingOutputStream.hash().asBytes());

    // Update the bimap
    fingerprintToContents.put(fingerprint, children);
    contentsToFingerprint.put(children, fingerprint);

    return fingerprint;
  }
}
