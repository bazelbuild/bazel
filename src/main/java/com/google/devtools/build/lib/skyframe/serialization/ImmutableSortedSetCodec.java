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

import com.google.common.collect.ImmutableSortedSet;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * {@link ObjectCodec} for {@link ImmutableSortedSet}. Comparator must be serializable, ideally a
 * registered constant.
 */
class ImmutableSortedSetCodec<E> implements ObjectCodec<ImmutableSortedSet<E>> {
  @SuppressWarnings("unchecked")
  @Override
  public Class<ImmutableSortedSet<E>> getEncodedClass() {
    return (Class<ImmutableSortedSet<E>>) (Class<?>) ImmutableSortedSet.class;
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableSortedSet<E> object, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serialize(object.comparator(), codedOut);
    codedOut.writeInt32NoTag(object.size());
    for (Object obj : object) {
      context.serialize(obj, codedOut);
    }
  }

  @Override
  public ImmutableSortedSet<E> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    ImmutableSortedSet.Builder<E> builder =
        ImmutableSortedSet.orderedBy(context.deserialize(codedIn));
    int size = codedIn.readInt32();
    for (int i = 0; i < size; i++) {
      builder.add(context.<E>deserialize(codedIn));
    }
    return builder.build();
  }
}
