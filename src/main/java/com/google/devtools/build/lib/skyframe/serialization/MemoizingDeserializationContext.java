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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;

/** Implementation that performs memoization. */
abstract class MemoizingDeserializationContext extends DeserializationContext {
  private final Memoizer.Deserializer deserializer = new Memoizer.Deserializer();

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
    deserializer.registerInitialValue(initialValue);
  }

  @Override
  final Object getMemoizedBackReference(int memoIndex) {
    return deserializer.getMemoized(memoIndex);
  }

  @Override
  final Object deserializeAndMaybeMemoize(ObjectCodec<?> codec, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return deserializer.deserialize(this, codec, codedIn);
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
  abstract Object deserializeAndMaybeHandleDeferredValues(
      ObjectCodec<?> codec, CodedInputStream codedIn) throws SerializationException, IOException;

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
      return codec.deserialize(this, codedIn);
    }
  }
}
