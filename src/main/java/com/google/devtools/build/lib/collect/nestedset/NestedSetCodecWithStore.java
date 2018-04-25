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

/**
 * Codec for {@link NestedSet} that uses the {@link NestedSetStore}.
 *
 * <p>Currently not used in favor of an @{@link AutoCodec}-ed NestedSet. Disabled by just not ending
 * in "Codec".
 */
public class NestedSetCodecWithStore<T> implements ObjectCodec<NestedSet<T>> {

  private final NestedSetStore nestedSetStore;

  /** Creates a NestedSetCodecWithStore that will use the given {@link NestedSetStore}. */
  public NestedSetCodecWithStore(NestedSetStore nestedSetStore) {
    this.nestedSetStore = nestedSetStore;
  }

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

    // If the NestedSet is a singleton, we serialize directly as an optimization.
    codedOut.writeBoolNoTag(obj.isSingleton());
    if (obj.isSingleton()) {
      context.serialize(obj.rawChildren(), codedOut);
      return;
    }

    ByteString fingerprint =
        nestedSetStore.computeFingerprintAndStore((Object[]) obj.rawChildren(), context);
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

    boolean isSingleton = codedIn.readBool();
    if (isSingleton) {
      T contents = context.deserialize(codedIn);
      return new NestedSet<T>(order, contents);
    }

    ByteString fingerprint = ByteString.copyFrom(codedIn.readByteArray());
    Object members = nestedSetStore.getContentsAndDeserialize(fingerprint, context);
    return new NestedSet<>(order, members);
  }
}
