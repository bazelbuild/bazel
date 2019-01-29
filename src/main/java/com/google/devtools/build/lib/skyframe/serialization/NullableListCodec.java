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

import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Encodes a list of elements, each of which may be null. Unless explicitly specified as the codec,
 * lists will only use this codec if they extend {@link AbstractList} or are one of {@link
 * #RANDOM_ACCESS_TYPE} or {@link #SEQUENTIAL_ACCESS_TYPE} (intended to capture unmodifiable
 * versions of normal lists).
 */
class NullableListCodec<T> implements ObjectCodec<List<T>> {
  @SuppressWarnings("unchecked")
  private static final Class<? extends List<?>> RANDOM_ACCESS_TYPE =
      (Class<? extends List<?>>) Collections.unmodifiableList(new ArrayList<>()).getClass();
  // Need linked list to get sequential access class type.
  @SuppressWarnings({"unchecked", "JdkObsolete"})
  private static final Class<? extends List<?>> SEQUENTIAL_ACCESS_TYPE =
      (Class<? extends List<?>>) Collections.unmodifiableList(new LinkedList<>()).getClass();

  // Needed due to generics / type erasure.
  @SuppressWarnings("unchecked")
  @Override
  public Class<List<T>> getEncodedClass() {
    // We return AbstractList here because List is an interface, so it will never be hit in the
    // ancestor traversal of a concrete class.
    return cast(AbstractList.class);
  }

  @SuppressWarnings("unchecked")
  @Override
  public List<Class<? extends List<T>>> additionalEncodedClasses() {
    return ImmutableList.of(cast(RANDOM_ACCESS_TYPE), cast(SEQUENTIAL_ACCESS_TYPE));
  }

  @Override
  public void serialize(SerializationContext context, List<T> list, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(list.size());
    for (T item : list) {
      context.serialize(item, codedOut);
    }
  }

  @Override
  public List<T> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }

    ArrayList<T> list = new ArrayList<>(length);
    for (int i = 0; i < length; i++) {
      list.add(context.deserialize(codedIn));
    }
    return maybeTransform(list);
  }

  protected List<T> maybeTransform(ArrayList<T> startingList) {
    return Collections.unmodifiableList(startingList);
  }

  @SuppressWarnings({"unchecked", "rawtypes"}) // Raw List.
  private Class<List<T>> cast(Class<? extends List> clazz) {
    return (Class<List<T>>) clazz;
  }
}
