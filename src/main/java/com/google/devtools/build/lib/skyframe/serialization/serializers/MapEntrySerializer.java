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
import com.google.common.collect.Maps;
import java.util.Map.Entry;

/** {@link Serializer} for {@link Entry}. */
class MapEntrySerializer extends Serializer<Entry<Object, Object>> {

  @Override
  public void write(Kryo kryo, Output output, Entry<Object, Object> entry) {
    kryo.writeClassAndObject(output, entry.getKey());
    kryo.writeClassAndObject(output, entry.getValue());
  }

  @Override
  public Entry<Object, Object> read(
      Kryo kryo, Input input, Class<Entry<Object, Object>> unusedType) {
    return Maps.immutableEntry(kryo.readClassAndObject(input), kryo.readClassAndObject(input));
  }

  static void registerSerializers(Kryo kryo) {
    MapEntrySerializer serializer = new MapEntrySerializer();
    kryo.register(Entry.class, serializer);
    kryo.register(Maps.immutableEntry(null, null).getClass(), serializer);
  }
}
