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

package com.google.devtools.build.lib.skyframe.serialization.serializers;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * A {@link Lists.ReverseList} Serializer.
 *
 * <p>Reverses the list before writing and then again when reading. This preserves the initial type
 * as there is no other way to obtain Guava's "hidden" reversed list types.
 */
abstract class ReverseListSerializer extends Serializer<List<Object>> {

  @Override
  public void write(Kryo kryo, Output output, List<Object> list) {
    List<Object> reversed = Lists.reverse(list);
    output.writeInt(reversed.size(), true);
    for (Object elt : reversed) {
      kryo.writeClassAndObject(output, elt);
    }
  }

  static void registerSerializers(Kryo kryo) {
    kryo.register(getReversedLinkedClass(), new ReverseList());
    kryo.register(getReversedArrayClass(), new RandomAccessReverseList());
  }

  @VisibleForTesting
  static @SuppressWarnings("rawtypes") Class<? extends List> getReversedLinkedClass() {
    return Lists.reverse(Lists.newLinkedList()).getClass();
  }

  @VisibleForTesting
  static @SuppressWarnings("rawtypes") Class<? extends List> getReversedArrayClass() {
    return Lists.reverse(Lists.newArrayList()).getClass();
  }

  /** A {@link Lists.ReverseList} implementation based on a {@link LinkedList}. */
  private static class ReverseList extends ReverseListSerializer {

    @Override
    public List<Object> read(Kryo kryo, Input input, Class<List<Object>> type) {
      LinkedList<Object> list = new LinkedList<>();
      int length = input.readInt(true);
      for (int i = 0; i < length; ++i) {
        list.add(kryo.readClassAndObject(input));
      }
      return Lists.reverse(list);
    }
  }

  /** A {@link Lists.ReverseList} implementation based on an {@link ArrayList}. */
  private static class RandomAccessReverseList extends ReverseListSerializer {

    @Override
    public List<Object> read(Kryo kryo, Input input, Class<List<Object>> type) {
      int length = input.readInt(true);
      ArrayList<Object> list = new ArrayList<>(length);
      for (int i = 0; i < length; ++i) {
        list.add(kryo.readClassAndObject(input));
      }
      return Lists.reverse(list);
    }
  }
}
