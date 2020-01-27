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

import com.google.common.collect.Sets;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.LinkedHashSet;

/** {@link ObjectCodec} for {@link LinkedHashSet}. */
class LinkedHashSetCodec<E> implements ObjectCodec<LinkedHashSet<E>> {

  @SuppressWarnings("unchecked")
  @Override
  public Class<LinkedHashSet<E>> getEncodedClass() {
    return (Class<LinkedHashSet<E>>) (Class<?>) LinkedHashSet.class;
  }

  @Override
  public void serialize(
      SerializationContext context, LinkedHashSet<E> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.size());
    for (Object object : obj) {
      context.serialize(object, codedOut);
    }
  }

  @Override
  public LinkedHashSet<E> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    LinkedHashSet<E> set = Sets.newLinkedHashSetWithExpectedSize(size);
    for (int i = 0; i < size; i++) {
      set.add(context.<E>deserialize(codedIn));
    }
    return set;
  }
}
