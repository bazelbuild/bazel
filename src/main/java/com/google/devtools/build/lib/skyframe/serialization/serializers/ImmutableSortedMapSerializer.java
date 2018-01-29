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
import com.google.common.collect.ImmutableSortedMap;
import java.util.Comparator;
import java.util.Map.Entry;

/** Custom Kryo serializer for {@link ImmutableSortedMap}. */
class ImmutableSortedMapSerializer extends Serializer<ImmutableSortedMap<Object, Object>> {

  @Override
  public void write(Kryo kryo, Output output, ImmutableSortedMap<Object, Object> map) {
    kryo.writeClassAndObject(output, map.comparator());
    output.writeInt(map.size(), true);
    for (Entry<Object, Object> elt : map.entrySet()) {
      kryo.writeClassAndObject(output, elt.getKey());
      kryo.writeClassAndObject(output, elt.getValue());
    }
  }

  @Override
  @SuppressWarnings({"unchecked", "rawtypes"})
  public ImmutableSortedMap<Object, Object> read(
      Kryo kryo, Input input, Class<ImmutableSortedMap<Object, Object>> type) {
    ImmutableSortedMap.Builder builder =
        ImmutableSortedMap.orderedBy((Comparator<Object>) kryo.readClassAndObject(input));
    int size = input.readInt(true);
    for (int i = 0; i < size; ++i) {
      builder.put(kryo.readClassAndObject(input), kryo.readClassAndObject(input));
    }
    return builder.build();
  }

  static void registerSerializers(Kryo kryo) {
    kryo.register(ImmutableSortedMap.class, new ImmutableSortedMapSerializer());
  }
}
