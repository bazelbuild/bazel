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
import com.google.common.collect.ImmutableMap;
import java.util.EnumMap;
import java.util.Map;

/** {@link Serializer} for {@link ImmutableMap}. */
class ImmutableMapSerializer extends Serializer<ImmutableMap<Object, Object>> {

  private ImmutableMapSerializer() {
    setImmutable(true);
  }

  @Override
  public void write(Kryo kryo, Output output, ImmutableMap<Object, Object> immutableMap) {
    output.writeInt(immutableMap.size(), true);
    for (Map.Entry<Object, Object> entry : immutableMap.entrySet()) {
      kryo.writeClassAndObject(output, entry.getKey());
      kryo.writeClassAndObject(output, entry.getValue());
    }
  }

  @Override
  public ImmutableMap<Object, Object> read(
      Kryo kryo, Input input, Class<ImmutableMap<Object, Object>> type) {
    ImmutableMap.Builder<Object, Object> builder = ImmutableMap.builder();
    int length = input.readInt(true);
    for (int i = 0; i < length; ++i) {
      builder.put(kryo.readClassAndObject(input), kryo.readClassAndObject(input));
    }
    return builder.build();
  }

  /** Registers serializers for {@link ImmutableMap}. */
  static void registerSerializers(Kryo kryo) {
    ImmutableMapSerializer serializer = new ImmutableMapSerializer();

    kryo.register(ImmutableMap.class, serializer);
    kryo.register(ImmutableMap.of().getClass(), serializer);

    Object o1 = new Object();
    Object o2 = new Object();

    kryo.register(ImmutableMap.of(o1, o1).getClass(), serializer);
    kryo.register(ImmutableMap.of(o1, o1, o2, o2).getClass(), serializer);

    Map<DummyEnum, Object> enumMap = new EnumMap<>(DummyEnum.class);
    for (DummyEnum e : DummyEnum.values()) {
      enumMap.put(e, o1);
    }

    kryo.register(ImmutableMap.copyOf(enumMap).getClass(), serializer);
  }

  private enum DummyEnum {
    VALUE1,
    VALUE2
  }
}
