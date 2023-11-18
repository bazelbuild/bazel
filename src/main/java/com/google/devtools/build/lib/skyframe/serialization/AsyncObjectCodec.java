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

/** {@link ObjectCodec} that uses only {@link AsyncDeserializationContext}. */
public abstract class AsyncObjectCodec<T> implements ObjectCodec<T> {

  @Override
  public final MemoizationStrategy getStrategy() {
    return MemoizationStrategy.MEMOIZE_BEFORE;
  }

  /** Adapter for synchronous contexts. */
  @Override
  public final T deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return deserializeAsync(context, codedIn);
  }

  /**
   * This has the same contract as {@link #deserialize}, but narrows the {@code context} API to
   * methods that are compatible with async deserialization.
   */
  public abstract T deserializeAsync(AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException;
}
