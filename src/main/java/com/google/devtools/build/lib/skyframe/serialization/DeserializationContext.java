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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.Deserializer;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry.CodecDescriptor;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import javax.annotation.CheckReturnValue;

/**
 * Stateful class for providing additional context to a single deserialization "session". This class
 * is thread-safe so long as {@link #deserializer} is null. If it is not null, this class is not
 * thread-safe and should only be accessed on a single thread for deserializing one serialized
 * object (that may contain other serialized objects inside it).
 */
public class DeserializationContext {
  private final ObjectCodecRegistry registry;
  private final ImmutableMap<Class<?>, Object> dependencies;
  private final Memoizer.Deserializer deserializer;

  private DeserializationContext(
      ObjectCodecRegistry registry,
      ImmutableMap<Class<?>, Object> dependencies,
      Deserializer deserializer) {
    this.registry = registry;
    this.dependencies = dependencies;
    this.deserializer = deserializer;
  }

  @VisibleForTesting
  public DeserializationContext(
      ObjectCodecRegistry registry, ImmutableMap<Class<?>, Object> dependencies) {
    this(registry, dependencies, /*deserializer=*/ null);
  }

  @VisibleForTesting
  public DeserializationContext(ImmutableMap<Class<?>, Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  // TODO(shahan): consider making codedIn a member of this class.
  @SuppressWarnings({"TypeParameterUnusedInFormals", "unchecked"})
  public <T> T deserialize(CodedInputStream codedIn) throws IOException, SerializationException {
    int tag = codedIn.readSInt32();
    if (tag == 0) {
      return null;
    }
    if (tag < 0) {
      // Subtract 1 to undo transformation from SerializationContext to avoid null.
      return (T) deserializer.getMemoized(-tag - 1);
    }
    T constant = (T) registry.maybeGetConstantByTag(tag);
    if (constant != null) {
      return constant;
    }
    CodecDescriptor codecDescriptor = registry.getCodecDescriptorByTag(tag);
    if (deserializer == null) {
      return (T) codecDescriptor.deserialize(this, codedIn);
    } else {
      return deserializer.deserialize(this, (ObjectCodec<T>) codecDescriptor.getCodec(), codedIn);
    }
  }

  /**
   * Register an initial value for the currently deserializing value, for use by child objects that
   * may have references to it.
   *
   * <p>This is a noop when memoization is disabled.
   */
  public <T> void registerInitialValue(T initialValue) {
    if (deserializer == null) {
      return;
    }
    deserializer.registerInitialValue(initialValue);
  }

  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return type.cast(dependencies.get(type));
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
   * Returns a memoizing {@link DeserializationContext}, as getMemoizingContext above. Unlike
   * getMemoizingContext, this method is not idempotent - the returned context will always be fresh.
   */
  public DeserializationContext getNewMemoizingContext() {
    return new DeserializationContext(this.registry, this.dependencies, new Deserializer());
  }

  public DeserializationContext getNewNonMemoizingContext() {
    return new DeserializationContext(this.registry, this.dependencies, null);
  }
}
