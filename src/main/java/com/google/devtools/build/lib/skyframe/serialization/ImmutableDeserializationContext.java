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
import com.google.protobuf.CodedInputStream;
import java.io.IOException;

/** An immutable deserialization context. */
public final class ImmutableDeserializationContext extends DeserializationContext {

  @VisibleForTesting
  public ImmutableDeserializationContext(
      ObjectCodecRegistry registry, ImmutableClassToInstanceMap<Object> dependencies) {
    super(registry, dependencies);
  }

  @VisibleForTesting
  public ImmutableDeserializationContext(ImmutableClassToInstanceMap<Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  @VisibleForTesting
  public ImmutableDeserializationContext() {
    this(ImmutableClassToInstanceMap.of());
  }

  @Override
  public ImmutableDeserializationContext getFreshContext() {
    return this;
  }

  @Override
  public void registerInitialValue(Object initialValue) {}

  @Override
  Object deserializeAndMaybeMemoize(ObjectCodec<?> codec, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return codec.deserialize(this, codedIn);
  }

  @Override
  Object getMemoizedBackReference(int memoIndex) {
    throw new UnsupportedOperationException(
        "The tag should never be less than 0 in the stateless case");
  }
}
