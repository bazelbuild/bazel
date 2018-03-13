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
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs.MemoizationPermission;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.util.function.Function;
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
  private final MemoizationPermission memoizationPermission;
  private final Memoizer.Deserializer deserializer;

  private DeserializationContext(
      ObjectCodecRegistry registry,
      ImmutableMap<Class<?>, Object> dependencies,
      MemoizationPermission memoizationPermission,
      Deserializer deserializer) {
    this.registry = registry;
    this.dependencies = dependencies;
    this.memoizationPermission = memoizationPermission;
    this.deserializer = deserializer;
  }

  @VisibleForTesting
  public DeserializationContext(
      ObjectCodecRegistry registry, ImmutableMap<Class<?>, Object> dependencies) {
    this(registry, dependencies, MemoizationPermission.ALLOWED, /*deserializer=*/ null);
  }

  @VisibleForTesting
  public DeserializationContext(ImmutableMap<Class<?>, Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  DeserializationContext disableMemoization() {
    Preconditions.checkState(
        memoizationPermission == MemoizationPermission.ALLOWED, "memoization already disabled");
    Preconditions.checkState(deserializer == null, "deserializer already present");
    return new DeserializationContext(
        registry, dependencies, MemoizationPermission.DISABLED, deserializer);
  }

  /**
   * Returns a {@link DeserializationContext} that will memoize values it encounters (using
   * reference equality), the inverse of the memoization performed by a {@link SerializationContext}
   * returned by {@link SerializationContext#newMemoizingContext}. The context returned here should
   * be used instead of the original: memoization may only occur when using the returned context.
   *
   * <p>A new {@link DeserializationContext} will be created for each call of this method, since it
   * is assumed that each codec that calls this method has data that it needs to inject into its
   * children that should mask its parents' injected data. Thus, over-eagerly calling this method
   * will reduce the effectiveness of memoization.
   */
  @CheckReturnValue
  public DeserializationContext newMemoizingContext(Object additionalData) {
    Preconditions.checkNotNull(additionalData);
    Preconditions.checkState(
        memoizationPermission == MemoizationPermission.ALLOWED, "memoization disabled");
    // TODO(janakr,brandjon): De we want to support a single context with data that gets pushed and
    // popped.
    return new DeserializationContext(
        this.registry, this.dependencies, memoizationPermission, new Deserializer(additionalData));
  }

  /**
   * Construct an initial value for the currently deserializing value from the additional data
   * stored in a memoizing deserializer. The additional data must have type {@code klass}. The given
   * {@code initialValueFunction} should be a hermetic function that does not interact with this
   * context or {@code codedIn in any way}: it should be a pure function of an element of type
   * {@link A}.
   */
  public <A, T> T makeInitialValue(Function<A, T> initialValueFunction, Class<A> klass) {
    return deserializer.makeInitialValue(initialValueFunction, klass);
  }

  // TODO(shahan): consider making codedIn a member of this class.
  @SuppressWarnings({"TypeParameterUnusedInFormals", "unchecked"})
  public <T> T deserialize(CodedInputStream codedIn) throws IOException, SerializationException {
    int tag = codedIn.readSInt32();
    if (tag == 0) {
      return null;
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

  @SuppressWarnings("unchecked")
  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return (T) dependencies.get(type);
  }
}
