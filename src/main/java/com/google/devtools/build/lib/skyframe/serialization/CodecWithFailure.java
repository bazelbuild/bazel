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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;

/**
 * A codec that throws an error whenever used. For usage in versions of Bazel that should never be
 * serialing the instances of the type T.
 */
public class CodecWithFailure<T> extends LeafObjectCodec<T> {

  private final String message;
  private final Class<?> clazz;

  public CodecWithFailure(Class<?> clazz, String message) {
    this.clazz = clazz;
    this.message = message;
  }

  @SuppressWarnings("unchecked")
  @Override
  public Class<? extends T> getEncodedClass() {
    return (Class<? extends T>) clazz;
  }

  @Override
  public void serialize(LeafSerializationContext context, T obj, CodedOutputStream codedOut)
      throws SerializationException {
    throw new SerializationException(message);
  }

  @Override
  public T deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException {
    throw new SerializationException(message);
  }
}
