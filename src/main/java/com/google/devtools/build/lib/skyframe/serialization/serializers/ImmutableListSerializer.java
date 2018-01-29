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
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Table;

/** A {@link Serializer} for {@link ImmutableList}. */
class ImmutableListSerializer extends Serializer<ImmutableList<Object>> {

  private ImmutableListSerializer() {
    setImmutable(true);
  }

  @Override
  public void write(Kryo kryo, Output output, ImmutableList<Object> object) {
    output.writeInt(object.size(), true);
    for (Object elm : object) {
      kryo.writeClassAndObject(output, elm);
    }
  }

  @Override
  public ImmutableList<Object> read(Kryo kryo, Input input, Class<ImmutableList<Object>> type) {
    int size = input.readInt(true);
    Object[] list = new Object[size];
    for (int i = 0; i < size; ++i) {
      list[i] = kryo.readClassAndObject(input);
    }
    return ImmutableList.copyOf(list);
  }

  /**
   * Creates a new {@link ImmutableListSerializer} and registers its serializer for the several
   * ImmutableList related classes.
   *
   * @param kryo the {@link Kryo} instance to set the serializer on
   */
  static void registerSerializers(Kryo kryo) {

    // ImmutableList (abstract class)
    //  +- RegularImmutableList
    //  |   RegularImmutableList
    //  +- SingletonImmutableList
    //  |   Optimized for List with only 1 element.
    //  +- SubList
    //  |   Representation for part of ImmutableList
    //  +- ReverseImmutableList
    //  |   For iterating in reverse order
    //  +- StringAsImmutableList
    //  |   Used by Lists#charactersOf
    //  +- Values (ImmutableTable values)
    //      Used by return value of #values() when there are multiple cells

    ImmutableListSerializer serializer = new ImmutableListSerializer();

    kryo.register(ImmutableList.class, serializer);

    // Note:
    //  Only registering above is good enough for serializing/deserializing.
    //  but if using Kryo#copy, following is required.

    kryo.register(ImmutableList.of().getClass(), serializer);
    kryo.register(ImmutableList.of(1).getClass(), serializer);
    kryo.register(ImmutableList.of(1, 2).getClass(), serializer);
    kryo.register(ImmutableList.of(1, 2, 3, 4).subList(1, 3).getClass(), serializer);
    kryo.register(ImmutableList.of(1, 2).reverse().getClass(), serializer);

    kryo.register(Lists.charactersOf("KryoRocks").getClass(), serializer);

    Table<Integer, Integer, Integer> baseTable = HashBasedTable.create();
    baseTable.put(1, 2, 3);
    baseTable.put(4, 5, 6);
    Table<Integer, Integer, Integer> table = ImmutableTable.copyOf(baseTable);
    kryo.register(table.values().getClass(), serializer);
  }
}
