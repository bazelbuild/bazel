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

/**
 * Implementation that supports sharing of sub-objects between objects.
 *
 * <p>This is a stub with the same behavior as the {@link MemoizingDeserializationContext}.
 */
// TODO: b/297857068 - complete this implementation
final class SharedValueDeserializationContext extends MemoizingDeserializationContext {
  @VisibleForTesting // private
  static SharedValueDeserializationContext createForTesting(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    return new SharedValueDeserializationContext(codecRegistry, dependencies);
  }

  private SharedValueDeserializationContext(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    super(codecRegistry, dependencies);
  }

  static Object deserializeWithSharedValues(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      ByteString bytes)
      throws SerializationException {
    return ObjectCodecs.deserializeStreamFully(
        bytes.newCodedInput(), new SharedValueDeserializationContext(codecRegistry, dependencies));
  }

  @Override
  public SharedValueDeserializationContext getFreshContext() {
    return new SharedValueDeserializationContext(getRegistry(), getDependencies());
  }

  @Override
  Object deserializeAndMaybeHandleDeferredValues(ObjectCodec<?> codec, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return codec.deserialize(this, codedIn);
  }
}
