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
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

/** A {@link Serializer} for {@link Class}. */
@SuppressWarnings("rawtypes")
class ClassSerializer extends Serializer<Class> {

  @Override
  public void write(Kryo unusedKryo, Output output, Class object) {
    output.writeAscii(object.getName());
  }

  @Override
  public Class read(Kryo unusedKryo, Input input, Class<Class> type) {
    try {
      return Class.forName(input.readString());
    } catch (ClassNotFoundException e) {
      throw new KryoException(e);
    }
  }

  static void registerSerializers(Kryo kryo) {
    kryo.register(Class.class, new ClassSerializer());
  }
}
