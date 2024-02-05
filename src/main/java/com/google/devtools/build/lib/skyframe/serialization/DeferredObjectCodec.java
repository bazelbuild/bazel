// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.util.function.Supplier;

/**
 * {@link ObjectCodec} that returns a continuation when deserializing.
 *
 * <p>The {@link AsyncDeserializationContext} can defer invoking of the continuation until all
 * asynchronous dependencies are resolved.
 */
public abstract class DeferredObjectCodec<T> implements ObjectCodec<T> {

  @Override
  public final MemoizationStrategy getStrategy() {
    return MemoizationStrategy.MEMOIZE_AFTER;
  }

  /** Implementation that adapts this codec for synchronous use. */
  @Override
  public final T deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return deserializeDeferred(context, codedIn).get();
  }

  /**
   * This differs from {@link #deserialize} by using the narrower {@link
   * AsyncDeserializationContext} and returning a {@link Supplier}.
   *
   * <p>This is used in cases where the deserialized object cannot even be constructed before the
   * children become available, which is common for immutable types.
   *
   * <p>Note that {@link Supplier} is invoked when all child instances have been provided, but that
   * may include partially formed child-instances. Use {@link
   * FlatDeserializationContext#deserializeFully} if the child-instances must be fully formed.
   */
  public abstract Supplier<T> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException;
}
