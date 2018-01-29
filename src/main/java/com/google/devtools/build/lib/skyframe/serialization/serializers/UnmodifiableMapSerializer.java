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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** {@link Serializer} for objects produced by {@link Collections#unmodifiableMap}. */
class UnmodifiableMapSerializer extends Serializer<Map<Object, Object>> {

  @Override
  public void write(Kryo kryo, Output output, Map<Object, Object> map) {
    output.writeInt(map.size(), true);
    for (Map.Entry<Object, Object> entry : map.entrySet()) {
      kryo.writeClassAndObject(output, entry.getKey());
      kryo.writeClassAndObject(output, entry.getValue());
    }
  }

  @Override
  public Map<Object, Object> read(Kryo kryo, Input input, Class<Map<Object, Object>> unusedType) {
    LinkedHashMap<Object, Object> map = new LinkedHashMap<>();
    int length = input.readInt(true);
    for (int i = 0; i < length; ++i) {
      map.put(kryo.readClassAndObject(input), kryo.readClassAndObject(input));
    }
    return Collections.unmodifiableMap(map);
  }

  static void registerSerializers(Kryo kryo) {
    kryo.register(
        Collections.unmodifiableMap(new LinkedHashMap<Object, Object>()).getClass(),
        new UnmodifiableMapSerializer());
  }
}
