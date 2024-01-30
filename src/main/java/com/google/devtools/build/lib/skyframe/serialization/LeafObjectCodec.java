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
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * A codec that directly deserializes from the {@link CodedInputStream}.
 *
 * <p>Since it never delegates to the {@link DeserializationContext}, it is asynchronous compatible.
 */
public abstract class LeafObjectCodec<T> implements ObjectCodec<T> {
  @Override
  public final void serialize(SerializationContext context, T obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    serialize((SerializationDependencyProvider) context, obj, codedOut);
  }

  /**
   * This has the same contract as {@link #serialize}, but may only depend on {@link
   * SerializationDependencyProvider} instead of the full {@link SerializationContext}.
   */
  public abstract void serialize(
      SerializationDependencyProvider dependencies, T obj, CodedOutputStream codedOut)
      throws SerializationException, IOException;

  @Override
  public final T deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return deserialize((SerializationDependencyProvider) context, codedIn);
  }

  /**
   * This has the same contract as {@link #deserialize}, but may only depend on {@link
   * SerializationDependencyProvider} instead of the full {@link DeserializationContext}.
   */
  public abstract T deserialize(
      SerializationDependencyProvider dependencies, CodedInputStream codedIn)
      throws SerializationException, IOException;
}
