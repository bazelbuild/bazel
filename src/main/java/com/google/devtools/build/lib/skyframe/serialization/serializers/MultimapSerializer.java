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
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import java.util.Map;
import java.util.function.Supplier;

/** {@link Serializer} for {@link MultiMap} subclasses. */
class MultimapSerializer<E, T extends Multimap<E, E>> extends Serializer<T> {

  private final Supplier<T> create;

  /**
   * Constructor.
   *
   * @param create reference to T.create
   */
  private MultimapSerializer(Supplier<T> create) {
    this.create = create;
  }

  @Override
  public void write(Kryo kryo, Output output, T multimap) {
    output.writeInt(multimap.size(), true);
    for (Map.Entry<E, E> entry : multimap.entries()) {
      kryo.writeClassAndObject(output, entry.getKey());
      kryo.writeClassAndObject(output, entry.getValue());
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public T read(Kryo kryo, Input input, Class<T> unusedType) {
    T multimap = create.get();
    kryo.reference(multimap);
    int size = input.readInt(true);
    for (int i = 0; i < size; ++i) {
      multimap.put((E) kryo.readClassAndObject(input), (E) kryo.readClassAndObject(input));
    }
    return multimap;
  }

  /** Registers serializers for {@link Multimap} subclasses. */
  static void registerSerializers(Kryo kryo) {
    kryo.register(ArrayListMultimap.class, new MultimapSerializer<>(ArrayListMultimap::create));
    kryo.register(HashMultimap.class, new MultimapSerializer<>(HashMultimap::create));
    kryo.register(LinkedHashMultimap.class, new MultimapSerializer<>(LinkedHashMultimap::create));
    kryo.register(LinkedListMultimap.class, new MultimapSerializer<>(LinkedListMultimap::create));
    kryo.register(TreeMultimap.class, new MultimapSerializer<>(TreeMultimap::create));
  }
}
