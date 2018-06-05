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

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
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
import java.util.concurrent.ExecutionException;

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
  /**
   * Used to preserve the invariant that if NestedSets inside two different objects are
   * reference-equal, they will continue to be reference-equal after deserialization.
   *
   * <p>Suppose NestedSet N is contained in objects A and B. If A is deserialized and then B is
   * deserialized, then when we create N inside B, we will use the version already created inside A.
   * This depends on the fact that NestedSets with the same underlying Object[] children and order
   * are equal, and that we have a cache of children Object[] that will contain N's children field
   * as long as it is in memory.
   *
   * <p>If A and B are created, then B is serialized and deserialized while A remains in memory, the
   * first serialization will put N into this interner, and so the deserialization will reuse it.
   */
  private final Cache<EqualsWrapper<T>, NestedSet<T>> interner =
      CacheBuilder.newBuilder().weakValues().build();

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
      codedOut.writeEnumNoTag(NestedSetSize.EMPTY.ordinal());
    } else if (obj.isSingleton()) {
      // If the NestedSet is a singleton, we serialize directly as an optimization.
      codedOut.writeEnumNoTag(NestedSetSize.SINGLETON.ordinal());
      context.serialize(obj.rawChildren(), codedOut);
    } else {
      codedOut.writeEnumNoTag(NestedSetSize.GROUP.ordinal());
      FingerprintComputationResult fingerprintComputationResult =
          nestedSetStore.computeFingerprintAndStore((Object[]) obj.rawChildren(), context);
      context.addFutureToBlockWritingOn(fingerprintComputationResult.writeStatus());
      codedOut.writeByteArrayNoTag(fingerprintComputationResult.fingerprint().toByteArray());
    }
    interner.put(new EqualsWrapper<>(obj), obj);
  }

  @Override
  public NestedSet<T> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    if (!SerializationConstants.shouldSerializeNestedSet) {
      // Don't perform NestedSet deserialization in testing
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    Order order = context.deserialize(codedIn);
    NestedSetSize nestedSetSize = NestedSetSize.values()[codedIn.readEnum()];
    switch (nestedSetSize) {
      case EMPTY:
        return NestedSetBuilder.emptySet(order);
      case SINGLETON:
        T contents = context.deserialize(codedIn);
        return intern(order, contents);
      case GROUP:
        ByteString fingerprint = ByteString.copyFrom(codedIn.readByteArray());
        Object members = nestedSetStore.getContentsAndDeserialize(fingerprint, context);
        return intern(order, members);
    }
    throw new IllegalStateException("NestedSet size " + nestedSetSize + " not known");
  }

  /**
   * Morally, NestedSets are compared using reference equality, to avoid the cost of unrolling them.
   * However, when deserializing NestedSet, we don't want to end up with two sets that "should" be
   * reference-equal, but are not. Since our codec implementation caches the underlying {@link
   * NestedSet#children} Object[], two nested sets that should be the same will have equal
   * underlying {@link NestedSet#children}, so we can use that for an equality check.
   *
   * <p>Note that singleton NestedSets' underlying children are not cached, but we must still
   * enforce equality for them. To do that, we use the #hashCode and #equals of the {@link
   * NestedSet#children}. When that field is an Object[], this is just identity hash code and
   * reference equality, but when it is something else (like an Artifact), we will do an actual
   * equality comparison. This may make some singleton NestedSets reference-equal where they were
   * not before. This should be ok, but isn't because Artifact does not properly implement equality
   * (it ignores the ArtifactOwner).
   *
   * <p>TODO(janakr): Fix this bug.
   */
  private NestedSet<T> intern(Order order, Object contents) {
    NestedSet<T> result = new NestedSet<>(order, contents);
    try {
      return interner.get(new EqualsWrapper<>(result), () -> result);
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }
  }

  private static final class EqualsWrapper<T> {
    private final NestedSet<T> nestedSet;

    private EqualsWrapper(NestedSet<T> nestedSet) {
      this.nestedSet = nestedSet;
    }

    @Override
    public int hashCode() {
      return 37 * nestedSet.getOrder().hashCode() + nestedSet.rawChildren().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof EqualsWrapper)) {
        return false;
      }
      NestedSet<?> thatSet = ((EqualsWrapper<?>) obj).nestedSet;
      return this.nestedSet.getOrder().equals(thatSet.getOrder())
          && this.nestedSet.rawChildren().equals(thatSet.rawChildren());
    }
  }
}
