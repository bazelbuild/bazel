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

import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.FingerprintComputationResult;
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

  private enum NestedSetSize {
    EMPTY, SINGLETON, GROUP
  }

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
    if (obj.isEmpty()) {
      // If the NestedSet is empty, it needs to be assigned to the EMPTY_CHILDREN constant on
      // deserialization.
      context.serialize(NestedSetSize.EMPTY, codedOut);
    } else if (obj.isSingleton()) {
      // If the NestedSet is a singleton, we serialize directly as an optimization.
      context.serialize(NestedSetSize.SINGLETON, codedOut);
      context.serialize(obj.rawChildren(), codedOut);
    } else {
      context.serialize(NestedSetSize.GROUP, codedOut);
      FingerprintComputationResult fingerprintComputationResult =
          nestedSetStore.computeFingerprintAndStore((Object[]) obj.rawChildren(), context);
      context.addFutureToBlockWritingOn(fingerprintComputationResult.writeStatus());
      codedOut.writeByteArrayNoTag(fingerprintComputationResult.fingerprint().toByteArray());
    }
  }

  @Override
  public NestedSet<T> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    if (!SerializationConstants.shouldSerializeNestedSet) {
      // Don't perform NestedSet deserialization in testing
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    Order order = context.deserialize(codedIn);
    NestedSetSize nestedSetSize = context.deserialize(codedIn);
    switch (nestedSetSize) {
      case EMPTY:
        return NestedSetBuilder.emptySet(order);
      case SINGLETON:
        T contents = context.deserialize(codedIn);
        return new NestedSet<>(order, contents);
      case GROUP:
        ByteString fingerprint = ByteString.copyFrom(codedIn.readByteArray());
        Object members = nestedSetStore.getContentsAndDeserialize(fingerprint, context);
        return new NestedSet<>(order, members);
    }
    throw new IllegalStateException("NestedSet size " + nestedSetSize + " not known");
  }
}
