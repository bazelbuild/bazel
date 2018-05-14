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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;

/**
 * Interface for Package deserialization.
 *
 * <p>Provides a layer of indirection for breaking circular dependencies.
 */
public interface PackageDeserializerInterface {

  /**
   * Deserializes a {@link Package} from {@code codedIn}. The inverse of {@link
   * PackageSerializer#serialize}.
   *
   * @param codedIn stream to read from
   * @return a new {@link Package} as read from {@code codedIn}
   * @throws IOException on failures reading from {@code codedIn}
   * @throws InterruptedException
   * @throws SerializationException on failures deserializing the input
   */
  Package deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws IOException, InterruptedException, SerializationException;
}
