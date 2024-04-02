// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.ForOverride;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * {@link DeserializationContext} that performs memoization, see {@link
 * MemoizingSerializationContext} for the protocol description.
 */
abstract class MemoizingDeserializationContext extends DeserializationContext {
  private final Int2ObjectOpenHashMap<Object> memoTable = new Int2ObjectOpenHashMap<>();
  private int tagForMemoizedBefore = -1;
  private final Deque<Object> memoizedBeforeStackForSanityChecking = new ArrayDeque<>();

  @VisibleForTesting // private
  static MemoizingDeserializationContext createForTesting(
      ObjectCodecRegistry registry, ImmutableClassToInstanceMap<Object> dependencies) {
    return new MemoizingDeserializationContextImpl(registry, dependencies);
  }

  MemoizingDeserializationContext(
      ObjectCodecRegistry registry, ImmutableClassToInstanceMap<Object> dependencies) {
    super(registry, dependencies);
  }

  static Object deserializeMemoized(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      ByteString bytes)
      throws SerializationException {
    return ObjectCodecs.deserializeStreamFully(
        bytes.newCodedInput(),
        new MemoizingDeserializationContextImpl(codecRegistry, dependencies));
  }

  static Object deserializeMemoized(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      byte[] bytes)
      throws SerializationException {
    return ObjectCodecs.deserializeStreamFully(
        CodedInputStream.newInstance(bytes),
        new MemoizingDeserializationContextImpl(codecRegistry, dependencies));
  }

  @Override
  public final void registerInitialValue(Object initialValue) {
    Preconditions.checkState(
        tagForMemoizedBefore != -1, "Not called with memoize before: %s", initialValue);
    int tag = tagForMemoizedBefore;
    tagForMemoizedBefore = -1;
    memoize(tag, initialValue);
    memoizedBeforeStackForSanityChecking.addLast(initialValue);
  }

  @Override
  final Object getMemoizedBackReference(int memoIndex) {
    return checkNotNull(memoTable.get(memoIndex), memoIndex);
  }

  @Override
  final Object deserializeAndMaybeMemoize(ObjectCodec<?> codec, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Preconditions.checkState(
        tagForMemoizedBefore == -1,
        "non-null memoized-before tag %s (%s)",
        tagForMemoizedBefore,
        codec);
    switch (codec.getStrategy()) {
      case MEMOIZE_BEFORE:
        return deserializeMemoBeforeContent(codec, codedIn);
      case MEMOIZE_AFTER:
        return deserializeMemoAfterContent(codec, codedIn);
    }
    throw new AssertionError("Unreachable (strategy=" + codec.getStrategy() + ")");
  }

  /**
   * Deserializes from {@code codedIn} using {@code codec}.
   *
   * <p>This extension point allows the implementation to optionally handle read futures and surface
   * {@link DeferredValue}s, which are possible for {@link SharedValueDeserializationContext}.
   *
   * <p>This can return either a deserialized value or a {@link DeferredValue}. A {@link
   * DeferredValue} is only possible for {@link SharedValueDeserializationContext}.
   */
  @ForOverride
  abstract Object deserializeAndMaybeHandleDeferredValues(
      ObjectCodec<?> codec, CodedInputStream codedIn) throws SerializationException, IOException;

  /**
   * Corresponds to MemoBeforeContent in the abstract grammar.
   *
   * <p>May return a deserialized value or a {@link ListenableFuture}. The {@link ListenableFuture}
   * is only possible for {@link SharedValueDeserializationContext}.
   */
  private final Object deserializeMemoBeforeContent(ObjectCodec<?> codec, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int tag = codedIn.readInt32();
    this.tagForMemoizedBefore = tag;
    // `codec` is never a `DeferredObjectCodec` because those are `MEMOIZE_AFTER` so this is always
    // the deserialized value instance and never a `DeferredValue`.
    Object value = deserializeAndMaybeHandleDeferredValues(codec, codedIn);
    Object initial = memoizedBeforeStackForSanityChecking.removeLast();
    if (value != initial) {
      // This indicates a bug in the particular codec subclass.
      throw new SerializationException(
          String.format(
              "codec did not return the initial instance: %s but was %s with codec %s",
              value, initial, codec));
    }

    Object combinedValue = combineValueWithReadFutures(value);
    if (combinedValue != value) {
      // If the combined value is different, it means that it is a ListenableFuture and there are
      // are read futures for this value. The (partial) value for `tag` will already be memoized by
      // `registerInitialValue` at this point.
      //
      // Any backreferences to the existing entry would be from cyclic children, which
      // tautologically need to tolerate incomplete values anyway. However, any subsequent
      // backreferences will observe the ListenableFuture and process it so that only complete
      // values are consumed.
      updateMemoEntry(tag, combinedValue);
      return combinedValue;
    }
    return value;
  }

  /**
   * Corresponds to MemoAfterContent in the abstract grammar.
   *
   * <p>May return either a deserialized value or a {@link ListenableFuture}. The {@link
   * ListenableFuture} is only possible for {@link SharedValueDeserializationContext}.
   */
  private final Object deserializeMemoAfterContent(ObjectCodec<?> codec, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Object value =
        combineValueWithReadFutures(deserializeAndMaybeHandleDeferredValues(codec, codedIn));
    int tag = codedIn.readInt32();
    // If deserializing the children caused the parent object itself to be deserialized due to
    // a cycle, then there's now a memo entry for the parent. Reuse that object, discarding
    // the one we were trying to construct here, so as to avoid creating duplicate objects in
    // the object graph.
    Object cyclicallyCreatedObject = memoTable.get(tag);
    if (cyclicallyCreatedObject != null) {
      return cyclicallyCreatedObject;
    }
    memoize(tag, value);
    return value;
  }

  /**
   * Incorporates read futures in the context together with {@code value}.
   *
   * <p>May return the deserialized value or a {@link ListenableFuture} that wraps the deserialized
   * value. The {@link ListenableFuture} is only possible for {@link
   * SharedValueDeserializationContext}.
   */
  @ForOverride
  abstract Object combineValueWithReadFutures(Object value);

  /**
   * Adds a new id → object maplet to the memo table.
   *
   * <p>It is an error if the value is already be present.
   */
  private final void memoize(int id, Object value) {
    Object prev = memoTable.put(id, checkNotNull(value));
    if (prev != null) { // Avoid boxing int with checkArgument.
      throw new IllegalArgumentException(
          String.format(
              "Tried to memoize id %s to object '%s', when it is already memoized to object"
                  + " '%s'",
              id, value, prev));
    }
  }

  private void updateMemoEntry(int id, Object newValue) {
    Object prev = memoTable.put(id, newValue);
    checkState(prev != null, "Tried to update id %s but there was no previous entry", id);
  }

  private static final class MemoizingDeserializationContextImpl
      extends MemoizingDeserializationContext {
    private MemoizingDeserializationContextImpl(
        ObjectCodecRegistry registry, ImmutableClassToInstanceMap<Object> dependencies) {
      super(registry, dependencies);
    }

    @Override
    public MemoizingDeserializationContext getFreshContext() {
      return new MemoizingDeserializationContextImpl(getRegistry(), getDependencies());
    }

    @Override
    Object deserializeAndMaybeHandleDeferredValues(ObjectCodec<?> codec, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return codec.safeCast(codec.deserialize(this, codedIn));
    }

    @Override
    @CanIgnoreReturnValue
    Object combineValueWithReadFutures(Object value) {
      return value;
    }
  }
}
