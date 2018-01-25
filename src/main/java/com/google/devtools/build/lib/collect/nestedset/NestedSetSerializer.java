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

package com.google.devtools.build.lib.collect.nestedset;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

/**
 * {@link Serializer} for {@link NestedSet}.
 *
 * <p>Needed to handle {@link NestedSet}'s sentinel values correctly.
 */
public class NestedSetSerializer extends Serializer<NestedSet<Object>> {

  @Override
  public void write(Kryo kryo, Output output, NestedSet<Object> nestedSet) {
    kryo.writeObject(output, nestedSet.getOrder());
    Object children = nestedSet.rawChildren();
    if (children == NestedSet.EMPTY_CHILDREN) {
      output.writeBoolean(false);
    } else {
      output.writeBoolean(true);
      kryo.writeClassAndObject(output, children);
    }
  }

  @Override
  public NestedSet<Object> read(Kryo kryo, Input input, Class<NestedSet<Object>> unusedType) {
    Order order = kryo.readObject(input, Order.class);
    if (input.readBoolean()) {
      return new NestedSet<>(order, kryo.readClassAndObject(input));
    } else {
      return new NestedSet<>(order, NestedSet.EMPTY_CHILDREN);
    }
  }

  public static void registerSerializers(Kryo kryo) {
    kryo.register(NestedSet.class, new NestedSetSerializer());
    kryo.register(Order.class);
    kryo.register(Object[].class);
  }
}
