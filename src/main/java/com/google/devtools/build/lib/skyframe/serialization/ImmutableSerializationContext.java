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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * An explicitly immutable implementation of {@link SerializationContext}.
 *
 * <p>Immutability makes this class thread safe.
 */
final class ImmutableSerializationContext extends SerializationContext {
  ImmutableSerializationContext(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    super(codecRegistry, dependencies);
  }

  @Override
  public ImmutableSerializationContext getFreshContext() {
    return this;
  }

  @Override
  public void addExplicitlyAllowedClass(Class<?> allowedClass) throws SerializationException {
    throw new SerializationException(
        "Cannot add explicitly allowed class %s without memoization: " + allowedClass);
  }

  @Override
  public <T> void checkClassExplicitlyAllowed(Class<T> allowedClass, T objectForDebugging)
      throws SerializationException {
    throw new SerializationException(
        "Cannot check explicitly allowed class "
            + allowedClass
            + " without memoization ("
            + objectForDebugging
            + ")");
  }

  @Override
  void serializeWithCodec(ObjectCodec<Object> codec, Object obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codec.serialize(this, obj, codedOut);
  }

  @Override
  public <T> void serializeLeaf(T obj, LeafObjectCodec<T> codec, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codec.serialize((LeafSerializationContext) this, obj, codedOut);
  }

  @Override
  boolean writeBackReferenceIfMemoized(Object obj, CodedOutputStream codedOut) {
    return false;
  }

  @Override
  public boolean isMemoizing() {
    return false;
  }
}
