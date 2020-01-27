// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Encodes a list of elements using a specified {@link ObjectCodec}. */
public class ImmutableListCodec<T> implements ObjectCodec<ImmutableList<T>> {

  private final ObjectCodec<T> codec;

  ImmutableListCodec(ObjectCodec<T> codec) {
    this.codec = codec;
  }

  @SuppressWarnings("unchecked")
  @Override
  public Class<ImmutableList<T>> getEncodedClass() {
    // Compiler doesn't like cast from Class<ImmutableList> -> Class<ImmutableList<T>>, but it
    // does allow what we see below. Type is lost at runtime anyway, so while gross this works.
    return (Class<ImmutableList<T>>) ((Class<?>) ImmutableList.class);
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableList<T> list, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(list.size());
    for (T item : list) {
      codec.serialize(context, item, codedOut);
    }
  }

  @Override
  public ImmutableList<T> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }

    ImmutableList.Builder<T> builder = ImmutableList.builder();
    for (int i = 0; i < length; i++) {
      builder.add(codec.deserialize(context, codedIn));
    }
    return builder.build();
  }
}
