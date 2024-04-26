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

import com.google.protobuf.CodedInputStream;
import java.io.IOException;

/**
 * Context provided to {@link LeafObjectCodec} implementations.
 *
 * <p>This context permits delegation only to other {@link LeafObjectCodec} instances and dependency
 * lookups.
 */
public interface LeafDeserializationContext extends SerializationDependencyProvider {
  /** Deserializes an object from {@code codedIn} using {@code codec}. */
  public <T> T deserializeLeaf(CodedInputStream codedIn, LeafObjectCodec<T> codec)
      throws IOException, SerializationException;
}
