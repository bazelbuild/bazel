// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.protobuf.LazyStringArrayList;
import com.google.protobuf.LazyStringList;
import com.google.protobuf.UnmodifiableLazyStringList;
import java.io.IOException;
import java.util.List;

/** Codec for {@link LazyStringList} so protobuf objects have proper type, not just a list. */
class LazyStringListCodec implements ObjectCodec<LazyStringList> {
  // If LazyStringArrayList.EMPTY ever goes away, just replace with our own empty one.
  private static final UnmodifiableLazyStringList EMPTY_UNMODIFIABLE =
      new UnmodifiableLazyStringList(LazyStringArrayList.EMPTY);

  @Override
  public Class<LazyStringList> getEncodedClass() {
    return LazyStringList.class;
  }

  @Override
  public List<Class<? extends LazyStringList>> additionalEncodedClasses() {
    return ImmutableList.of(UnmodifiableLazyStringList.class, LazyStringArrayList.class);
  }

  @Override
  public void serialize(
      SerializationContext context, LazyStringList obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.size());
    for (String item : obj) {
      context.serialize(item, codedOut);
    }
  }

  @Override
  public UnmodifiableLazyStringList deserialize(
      DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    if (size == 0) {
      return EMPTY_UNMODIFIABLE;
    }
    LazyStringArrayList list = new LazyStringArrayList(size);
    for (int i = 0; i < size; i++) {
      list.add((String) context.deserialize(codedIn));
    }
    return new UnmodifiableLazyStringList(list);
  }
}
