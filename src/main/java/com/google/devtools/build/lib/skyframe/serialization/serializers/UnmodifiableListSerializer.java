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
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/** A {@link Serializer} for objects produced by {@link Collections#unmodifiableList}. */
class UnmodifiableListSerializer extends Serializer<List<Object>> {

  private static final Class<?> RANDOM_ACCESS_TYPE =
      Collections.unmodifiableList(new ArrayList<Object>()).getClass();
  private static final Class<?> SEQUENTIAL_ACCESS_TYPE =
      Collections.unmodifiableList(new LinkedList<Object>()).getClass();

  @Override
  public void write(Kryo kryo, Output output, List<Object> object) {
    output.writeInt(object.size(), true);
    for (Object elt : object) {
      kryo.writeClassAndObject(output, elt);
    }
  }

  @Override
  public List<Object> read(Kryo kryo, Input input, Class<List<Object>> type) {
    int size = input.readInt(true);
    List<Object> list = null;
    if (type.equals(RANDOM_ACCESS_TYPE)) {
      list = new ArrayList<>(size);
    } else if (type.equals(SEQUENTIAL_ACCESS_TYPE)) {
      list = new LinkedList<>();
    } else {
      throw new IllegalArgumentException(
          "UnmodifiableListSerializer.read called with type: " + type);
    }
    for (int i = 0; i < size; ++i) {
      list.add(kryo.readClassAndObject(input));
    }
    return Collections.unmodifiableList(list);
  }

  static void registerSerializers(Kryo kryo) {
    UnmodifiableListSerializer serializer = new UnmodifiableListSerializer();
    kryo.register(RANDOM_ACCESS_TYPE, serializer);
    kryo.register(SEQUENTIAL_ACCESS_TYPE, serializer);
  }
}
