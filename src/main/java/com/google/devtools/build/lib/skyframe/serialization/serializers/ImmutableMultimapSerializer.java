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
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSetMultimap;
import java.util.Collection;
import java.util.Map;

/** {@link Serializer} for {@link ImmutableMultimap}. */
class ImmutableMultimapSerializer extends Serializer<ImmutableMultimap<Object, Object>> {

  private ImmutableMultimapSerializer() {
    setImmutable(true);
  }

  @Override
  public void write(Kryo kryo, Output output, ImmutableMultimap<Object, Object> immutableMultiMap) {
    kryo.writeObject(output, ImmutableMap.copyOf(immutableMultiMap.asMap()));
  }

  @Override
  public ImmutableMultimap<Object, Object> read(
      Kryo kryo, Input input, Class<ImmutableMultimap<Object, Object>> type) {
    ImmutableMultimap.Builder<Object, Object> builder;
    if (type.equals(ImmutableListMultimap.class)) {
      builder = ImmutableListMultimap.builder();
    } else if (type.equals(ImmutableSetMultimap.class)) {
      builder = ImmutableSetMultimap.builder();
    } else {
      builder = ImmutableMultimap.builder();
    }

    @SuppressWarnings("unchecked")
    Map<Object, Collection<Object>> map = kryo.readObject(input, ImmutableMap.class);
    for (Map.Entry<Object, Collection<Object>> entry : map.entrySet()) {
      builder.putAll(entry.getKey(), entry.getValue());
    }
    return builder.build();
  }

  static void registerSerializers(Kryo kryo) {
    ImmutableMultimapSerializer serializer = new ImmutableMultimapSerializer();

    // ImmutableMultimap (abstract class)
    //  +- EmptyImmutableListMultimap
    //  +- ImmutableListMultimap
    //  +- EmptyImmutableSetMultimap
    //  +- ImmutableSetMultimap

    kryo.register(ImmutableMultimap.class, serializer);
    kryo.register(ImmutableListMultimap.of().getClass(), serializer);
    kryo.register(ImmutableListMultimap.of("A", "B").getClass(), serializer);
    kryo.register(ImmutableSetMultimap.of().getClass(), serializer);
    kryo.register(ImmutableSetMultimap.of("A", "B").getClass(), serializer);
  }
}
