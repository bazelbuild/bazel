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
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.Serializer;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs.MemoizationPermission;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException.NoCodecException;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;

/**
 * Stateful class for providing additional context to a single serialization "session". This class
 * is thread-safe so long as {@link #serializer} is null. If it is not null, this class is not
 * thread-safe and should only be accessed on a single thread for serializing one object (that may
 * involve serializing other objects contained in it).
 */
public class SerializationContext {
  private final ObjectCodecRegistry registry;
  private final ImmutableMap<Class<?>, Object> dependencies;
  private final MemoizationPermission memoizationPermission;
  @Nullable private final Memoizer.Serializer serializer;

  private SerializationContext(
      ObjectCodecRegistry registry,
      ImmutableMap<Class<?>, Object> dependencies,
      MemoizationPermission memoizationPermission,
      @Nullable Serializer serializer) {
    this.registry = registry;
    this.dependencies = dependencies;
    this.serializer = serializer;
    this.memoizationPermission = memoizationPermission;
    Preconditions.checkState(
        serializer == null || memoizationPermission == MemoizationPermission.ALLOWED);
  }

  @VisibleForTesting
  public SerializationContext(
      ObjectCodecRegistry registry, ImmutableMap<Class<?>, Object> dependencies) {
    this(registry, dependencies, MemoizationPermission.ALLOWED, /*serializer=*/ null);
  }

  @VisibleForTesting
  public SerializationContext(ImmutableMap<Class<?>, Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  SerializationContext disableMemoization() {
    Preconditions.checkState(
        memoizationPermission == MemoizationPermission.ALLOWED, "memoization already disabled");
    Preconditions.checkState(serializer == null, "serializer already present");
    return new SerializationContext(
        registry, dependencies, MemoizationPermission.DISABLED, serializer);
  }

  /**
   * Returns a {@link SerializationContext} that will memoize values it encounters (using reference
   * equality) in a new memoization table. The returned context should be used instead of the
   * original: memoization may only occur when using the returned context. Calls must be in pairs
   * with {@link DeserializationContext#getMemoizingContext} in the corresponding deserialization
   * code.
   *
   * <p>This method is idempotent: calling it on an already memoizing context will return the same
   * context.
   */
  @CheckReturnValue
  public SerializationContext getMemoizingContext() {
    Preconditions.checkState(
        memoizationPermission == MemoizationPermission.ALLOWED, "memoization disabled");
    if (serializer != null) {
      return this;
    }
    return new SerializationContext(
        this.registry, this.dependencies, memoizationPermission, new Memoizer.Serializer());
  }

  private boolean writeNullOrConstant(@Nullable Object object, CodedOutputStream codedOut)
      throws IOException {
    if (object == null) {
      codedOut.writeSInt32NoTag(0);
      return true;
    }
    Integer tag = registry.maybeGetTagForConstant(object);
    if (tag != null) {
      codedOut.writeSInt32NoTag(tag);
      return true;
    }
    return false;
  }

  @Nullable
  private ObjectCodecRegistry.CodecDescriptor recordAndGetDescriptorIfNotConstantOrNull(
      @Nullable Object object, CodedOutputStream codedOut) throws IOException, NoCodecException {
    if (writeNullOrConstant(object, codedOut)) {
      return null;
    }
    ObjectCodecRegistry.CodecDescriptor descriptor = registry.getCodecDescriptor(object.getClass());
    codedOut.writeSInt32NoTag(descriptor.getTag());
    return descriptor;
  }

  // TODO(shahan): consider making codedOut a member of this class.
  public void serialize(Object object, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    ObjectCodecRegistry.CodecDescriptor descriptor =
        recordAndGetDescriptorIfNotConstantOrNull(object, codedOut);
    if (descriptor != null) {
      if (serializer == null) {
        descriptor.serialize(this, object, codedOut);
      } else {
        @SuppressWarnings("unchecked")
        ObjectCodec<Object> castCodec = (ObjectCodec<Object>) descriptor.getCodec();
        serializer.serialize(this, object, castCodec, codedOut);
      }
    }
    }

  @SuppressWarnings("unchecked")
  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return (T) dependencies.get(type);
  }
}
