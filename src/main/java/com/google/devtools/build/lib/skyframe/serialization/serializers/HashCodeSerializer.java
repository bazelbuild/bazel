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
import com.google.common.hash.HashCode;

class HashCodeSerializer extends Serializer<HashCode> {

  @Override
  public void write(Kryo unusedKryo, Output output, HashCode hashCode) {
    byte[] bytes = hashCode.asBytes();
    output.writeInt(bytes.length, true);
    output.write(bytes);
  }

  @Override
  public HashCode read(Kryo kryo, Input input, Class<HashCode> unusedType) {
    return HashCode.fromBytes(input.readBytes(input.readInt(true)));
  }

  static void registerSerializers(Kryo kryo) {
    HashCodeSerializer serializer = new HashCodeSerializer();
    kryo.register(HashCode.class, serializer);
    kryo.register(HashCode.fromInt(0).getClass(), serializer);
    kryo.register(HashCode.fromLong(0).getClass(), serializer);
    kryo.register(HashCode.fromBytes(new byte[1]).getClass(), serializer);
  }
}
