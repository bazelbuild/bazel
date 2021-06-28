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

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.FingerprintComputationResult;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Codec for {@link NestedSet} that uses the {@link NestedSetStore}. */
public class NestedSetCodecWithStore implements ObjectCodec<NestedSet<?>> {

  private enum NestedSetSize {
    EMPTY, // distinguished empty node; size = 0, depth = 0
    LEAF, // a single element; size = 1, depth = 1
    NONLEAF // more than one element; size > 1, depth > 1
  }

  private static final FutureCallback<Void> CRASH_TERMINATING_CALLBACK =
      new FutureCallback<Void>() {
        @Override
        public void onSuccess(@Nullable Void result) {
          // Do nothing.
        }

        @Override
        public void onFailure(Throwable t) {
          BugReport.handleCrash(t);
        }
      };

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
  private final Cache<EqualsWrapper, NestedSet<?>> interner =
      Caffeine.newBuilder().weakValues().build();

  /** Creates a NestedSetCodecWithStore that will use the given {@link NestedSetStore}. */
  public NestedSetCodecWithStore(NestedSetStore nestedSetStore) {
    this.nestedSetStore = nestedSetStore;
  }

  @SuppressWarnings("unchecked")
  @Override
  public Class<NestedSet<?>> getEncodedClass() {
    // Compiler doesn't like cast from Class<NestedSet> -> Class<NestedSet<T>>, but it
    // does allow what we see below. Type is lost at runtime anyway, so while gross this works.
    return (Class<NestedSet<?>>) ((Class<?>) NestedSet.class);
  }

  @Override
  public void serialize(SerializationContext context, NestedSet<?> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serialize(obj.getOrder(), codedOut);
    if (obj.isEmpty()) {
      // If the NestedSet is empty, it needs to be assigned to the EMPTY_CHILDREN constant on
      // deserialization.
      codedOut.writeEnumNoTag(NestedSetSize.EMPTY.ordinal());
    } else if (obj.isSingleton()) {
      // If the NestedSet is a singleton, we serialize directly as an optimization.
      codedOut.writeEnumNoTag(NestedSetSize.LEAF.ordinal());
      context.serialize(obj.getChildren(), codedOut);
    } else {
      codedOut.writeEnumNoTag(NestedSetSize.NONLEAF.ordinal());
      context.serialize(obj.getApproxDepth(), codedOut);
      FingerprintComputationResult fingerprintComputationResult =
          nestedSetStore.computeFingerprintAndStore((Object[]) obj.getChildren(), context);
      context.addFutureToBlockWritingOn(
          fingerprintComputationResult.writeStatus(), CRASH_TERMINATING_CALLBACK);
      codedOut.writeByteArrayNoTag(fingerprintComputationResult.fingerprint().toByteArray());
    }
    interner.put(new EqualsWrapper(obj), obj);
  }

  @Override
  public NestedSet<?> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Order order = context.deserialize(codedIn);
    NestedSetSize nestedSetSize = NestedSetSize.values()[codedIn.readEnum()];
    switch (nestedSetSize) {
      case EMPTY:
        return NestedSetBuilder.emptySet(order);
      case LEAF:
        Object contents = context.deserialize(codedIn);
        return intern(order, 1, contents);
      case NONLEAF:
        int depth = context.deserialize(codedIn);
        ByteString fingerprint = ByteString.copyFrom(codedIn.readByteArray());
        return intern(order, depth, nestedSetStore.getContentsAndDeserialize(fingerprint, context));
    }
    throw new IllegalStateException("NestedSet size " + nestedSetSize + " not known");
  }

  /**
   * Morally, NestedSets are compared using reference equality, to avoid the cost of unrolling them.
   * However, when deserializing NestedSet, we don't want to end up with two sets that "should" be
   * reference-equal, but are not. Since our codec implementation caches the underlying {@link
   * NestedSet#getChildren()} Object[], two nested sets that should be the same will have equal
   * underlying {@link NestedSet#getChildren()}, so we can use that for an equality check.
   *
   * <p>We also would like to prevent the existence of two equal NestedSets in a single JVM, in
   * which one NestedSet contains an Object[] and the other contains a ListenableFuture<Object[]>
   * for the same contents. This can happen if a NestedSet is serialized, and then deserialized with
   * a call to storage for those contents. In order to guarantee that only one NestedSet exists for
   * a given Object[], the interner checks the doneness of any ListenableFuture, and if done, holds
   * on only to the "materialized" NestedSet, with the Object[] as its children object.
   *
   * <p>Note that singleton NestedSets' underlying children are not cached, but we must still
   * enforce equality for them. To do that, we use the #hashCode and #equals of the {@link
   * NestedSet#getChildren()}. When that field is an Object[], this is just identity hash code and
   * reference equality, but when it is something else (like an Artifact), we will do an actual
   * equality comparison. This may make some singleton NestedSets reference-equal where they were
   * not before. This should be ok as long as the contained object properly implements equality.
   */
  @SuppressWarnings("unchecked")
  private NestedSet<?> intern(Order order, int depth, Object contents) {
    NestedSet<?> result;
    if (contents instanceof ListenableFuture) {
      result = NestedSet.withFuture(order, depth, (ListenableFuture<Object[]>) contents);
    } else {
      result = NestedSet.forDeserialization(order, depth, contents);
    }
    return interner.get(new EqualsWrapper(result), unused -> result);
  }

  private static final class EqualsWrapper {
    private final Order order;
    private final Object children;

    private EqualsWrapper(NestedSet<?> nestedSet) {
      // Unwrap the fields we need so that we don't strongly retain the NestedSet.
      this.order = nestedSet.getOrder();
      this.children = nestedSet.children;
    }

    @Override
    public int hashCode() {
      int childrenHashCode;
      if (children instanceof ListenableFuture && ((ListenableFuture<?>) children).isDone()) {
        try {
          childrenHashCode = Futures.getDone((ListenableFuture<?>) children).hashCode();
        } catch (ExecutionException e) {
          // If the future failed, we can treat it as unequal to all non-future NestedSet instances
          // (using the hashCode of the Future object) and hide the exception until the NestedSet is
          // truly needed (i.e. unrolled). Note that NestedSetStore already attaches a listener to
          // this future that sends a bug report if it fails.
          childrenHashCode = children.hashCode();
        }
      } else {
        childrenHashCode = children.hashCode();
      }

      return 37 * order.hashCode() + childrenHashCode;
    }

    private static boolean deserializingAndMaterializedSetsAreEqual(
        Object[] contents, ListenableFuture<Object[]> contentsFuture) {
      if (!contentsFuture.isDone()) {
        return false;
      }

      try {
        return Futures.getDone(contentsFuture) == contents;
      } catch (ExecutionException e) {
        return false; // Treat a failure to fetch as unequal to a non-future NestedSet.
      }
    }

    @SuppressWarnings("unchecked")
    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof EqualsWrapper)) {
        return false;
      }

      // Both sets contain Object[] or both sets contain ListenableFuture<Object[]>
      EqualsWrapper that = (EqualsWrapper) obj;
      if (this.order.equals(that.order) && this.children.equals(that.children)) {
        return true;
      }

      // One set contains Object[], while the other contains ListenableFuture<Object[]>
      if (this.children instanceof ListenableFuture && that.children instanceof Object[]) {
        return deserializingAndMaterializedSetsAreEqual(
            (Object[]) that.children, (ListenableFuture<Object[]>) this.children);
      } else if (that.children instanceof ListenableFuture && this.children instanceof Object[]) {
        return deserializingAndMaterializedSetsAreEqual(
            (Object[]) this.children, (ListenableFuture<Object[]>) that.children);
      } else {
        return false;
      }
    }
  }
}
