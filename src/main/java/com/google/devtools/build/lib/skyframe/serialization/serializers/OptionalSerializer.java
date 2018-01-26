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
import com.google.common.base.Optional;

/** {@link Serializer} for {@link Optional}. */
class OptionalSerializer extends Serializer<Optional<Object>> {

  @Override
  public void write(Kryo kryo, Output output, Optional<Object> optional) {
    kryo.writeClassAndObject(output, optional.orNull());
  }

  @Override
  public Optional<Object> read(Kryo kryo, Input input, Class<Optional<Object>> unusedType) {
    return Optional.fromNullable(kryo.readClassAndObject(input));
  }

  static void registerSerializers(Kryo kryo) {
    OptionalSerializer serializer = new OptionalSerializer();
    kryo.register(Optional.class, serializer);
    kryo.register(Optional.absent().getClass(), serializer);
    kryo.register(Optional.of(0).getClass(), serializer);
  }
}
