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

/** Codec variant that interns the deserialization result. */
public abstract class InterningObjectCodec<T> implements ObjectCodec<T> {

  @Override
  public final MemoizationStrategy getStrategy() {
    // There is no fixed reference to an interned object until after it has been constructed and
    // passes through the interner. Therefore this is always MEMOIZE_AFTER.
    return MemoizationStrategy.MEMOIZE_AFTER;
  }

  /**
   * Adapter for synchronous contexts.
   *
   * <p>Deserializes using {@link #deserializeInterned} then calls {@link #intern} on the result.
   */
  @Override
  public final T deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return intern(deserializeInterned(context, codedIn));
  }

  /** Performs the deserialization work. */
  public abstract T deserializeInterned(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException;

  /** Interns the result of {@link #deserializeInterned}. */
  public abstract T intern(T value);
}
