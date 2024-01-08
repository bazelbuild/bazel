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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec.MemoizationStrategy;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry.CodecDescriptor;
import com.google.errorprone.annotations.CheckReturnValue;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Stateful class for providing additional context to a single deserialization "session". This class
 * is thread-safe so long as {@link #deserializer} is null. If it is not null, this class is not
 * thread-safe and should only be accessed on a single thread for deserializing one serialized
 * object (that may contain other serialized objects inside it).
 */
public class DeserializationContext implements AsyncDeserializationContext {
  private final ObjectCodecRegistry registry;
  private final ImmutableClassToInstanceMap<Object> dependencies;
  @Nullable private final Memoizer.Deserializer deserializer;

  private DeserializationContext(
      ObjectCodecRegistry registry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Memoizer.Deserializer deserializer) {
    this.registry = registry;
    this.dependencies = dependencies;
    this.deserializer = deserializer;
  }

  @VisibleForTesting
  public DeserializationContext(
      ObjectCodecRegistry registry, ImmutableClassToInstanceMap<Object> dependencies) {
    this(registry, dependencies, /*deserializer=*/ null);
  }

  @VisibleForTesting
  public DeserializationContext(ImmutableClassToInstanceMap<Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  @Nullable
  @SuppressWarnings({"TypeParameterUnusedInFormals"})
  public <T> T deserialize(CodedInputStream codedIn) throws IOException, SerializationException {
    return deserializeInternal(codedIn, /*customMemoizationStrategy=*/ null);
  }

  /**
   * Deserializes into {@code obj} using {@code setter}.
   *
   * <p>This allows custom processing of the deserialized object.
   */
  @Override
  public <T> void deserialize(CodedInputStream codedIn, T obj, FieldSetter<? super T> setter)
      throws IOException, SerializationException {
    Object value = deserializeInternal(codedIn, /* customMemoizationStrategy= */ null);
    if (value == null) {
      return;
    }
    setter.set(obj, value);
  }

  @Override
  public void deserialize(CodedInputStream codedIn, Object obj, long offset)
      throws IOException, SerializationException {
    Object value = deserializeInternal(codedIn, /* customMemoizationStrategy= */ null);
    if (value == null) {
      return;
    }
    unsafe().putObject(obj, offset, value);
  }

  @Override
  public void deserialize(CodedInputStream codedIn, Object obj, long offset, Runnable done)
      throws IOException, SerializationException {
    deserialize(codedIn, obj, offset);
    done.run();
  }

  @Override
  public void deserializeFully(CodedInputStream codedIn, Object obj, long offset)
      throws IOException, SerializationException {
    // This method is identical to the call below in the synchronous implementation.
    deserialize(codedIn, obj, offset);
  }

  @Override
  public <T> void deserializeFully(CodedInputStream codedIn, T obj, FieldSetter<? super T> setter)
      throws IOException, SerializationException {
    // This method is identical to the call below in the synchronous implementation.
    deserialize(codedIn, obj, setter);
  }

  @Override
  public void deserializeFully(CodedInputStream codedIn, Object obj, long offset, Runnable done)
      throws IOException, SerializationException {
    // This method is identical to the call below in the synchronous implementation.
    deserialize(codedIn, obj, offset, done);
  }

  @Nullable
  @SuppressWarnings({"TypeParameterUnusedInFormals"})
  public <T> T deserializeWithAdHocMemoizationStrategy(
      CodedInputStream codedIn, MemoizationStrategy memoizationStrategy)
      throws IOException, SerializationException {
    return deserializeInternal(codedIn, memoizationStrategy);
  }

  @Nullable
  @SuppressWarnings({"TypeParameterUnusedInFormals", "unchecked"})
  private <T> T deserializeInternal(
      CodedInputStream codedIn, @Nullable MemoizationStrategy customMemoizationStrategy)
      throws IOException, SerializationException {
    int tag = codedIn.readSInt32();
    if (tag == 0) {
      return null;
    }
    if (tag < 0) {
      // Subtract 1 to undo transformation from SerializationContext to avoid null.
      return (T) deserializer.getMemoized(-tag - 1); // unchecked cast
    }
    T constant = (T) registry.maybeGetConstantByTag(tag);
    if (constant != null) {
      return constant;
    }
    CodecDescriptor codecDescriptor = registry.getCodecDescriptorByTag(tag);
    if (deserializer == null) {
      return (T) codecDescriptor.deserialize(this, codedIn); // unchecked cast
    } else {
      @SuppressWarnings("unchecked")
      ObjectCodec<T> castCodec = (ObjectCodec<T>) codecDescriptor.getCodec();
      MemoizationStrategy memoizationStrategy =
          customMemoizationStrategy != null ? customMemoizationStrategy : castCodec.getStrategy();
      return deserializer.deserialize(this, castCodec, memoizationStrategy, codedIn);
    }
  }

  @Override
  public void registerInitialValue(Object initialValue) {
    if (deserializer != null) {
      deserializer.registerInitialValue(initialValue);
    }
  }

  @Override
  public <T> T getDependency(Class<T> type) {
    return checkNotNull(dependencies.getInstance(type), "Missing dependency of type %s", type);
  }

  /**
   * Returns a {@link DeserializationContext} that will memoize values it encounters (using
   * reference equality), the inverse of the memoization performed by a {@link SerializationContext}
   * returned by {@link SerializationContext#getMemoizingContext}. The context returned here should
   * be used instead of the original: memoization may only occur when using the returned context.
   *
   * <p>This method is idempotent: calling it on an already memoizing context will return the same
   * context.
   */
  @CheckReturnValue
  public DeserializationContext getMemoizingContext() {
    if (deserializer != null) {
      return this;
    }
    return getNewMemoizingContext();
  }

  /**
   * Returns a new memoizing {@link DeserializationContext}, as {@link #getMemoizingContext}. Unlike
   * {@link #getMemoizingContext}, this method is not idempotent - the returned context will always
   * be fresh.
   */
  public DeserializationContext getNewMemoizingContext() {
    return new DeserializationContext(registry, dependencies, new Memoizer.Deserializer());
  }

  /**
   * Returns a new {@link DeserializationContext} mostly identical to this one, but with a
   * dependency map composed by applying overrides to this context's dependencies.
   *
   * <p>The given {@code dependencyOverrides} may contain keys already present (in which case the
   * dependency is replaced) or new keys (in which case the dependency is added).
   *
   * <p>Must only be called on a base context (no memoization state), since changing dependencies
   * may change deserialization semantics.
   */
  @CheckReturnValue
  public DeserializationContext withDependencyOverrides(ClassToInstanceMap<?> dependencyOverrides) {
    checkState(deserializer == null, "Must only be called on base DeserializationContext");
    return new DeserializationContext(
        registry,
        ImmutableClassToInstanceMap.builder()
            .putAll(Maps.filterKeys(dependencies, k -> !dependencyOverrides.containsKey(k)))
            .putAll(dependencyOverrides)
            .build(),
        /*deserializer=*/ null);
  }
}
