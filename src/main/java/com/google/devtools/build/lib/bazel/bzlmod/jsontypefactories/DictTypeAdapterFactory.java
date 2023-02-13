// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod.jsontypefactories;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.gson.Gson;
import com.google.gson.TypeAdapter;
import com.google.gson.TypeAdapterFactory;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.inject.Inject;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Mutability;

/**
 * Serializes and deserializes {@link ImmutableMap} instances using a {@link HashMap} as an
 * intermediary.
 */
final public class DictTypeAdapterFactory implements TypeAdapterFactory {

  @Inject
  public DictTypeAdapterFactory() {}

  @SuppressWarnings("unchecked")
  @Override
  public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> typeToken) {
    Type type = typeToken.getType();
    if (typeToken.getRawType() != Dict.class || !(type instanceof ParameterizedType)) {
      return null;
    }

    com.google.common.reflect.TypeToken<Dict<?, ?>> betterToken =
        (com.google.common.reflect.TypeToken<Dict<?, ?>>)
            com.google.common.reflect.TypeToken.of(typeToken.getType());
    final TypeAdapter<Map<?, ?>> dictAdapter =
        (TypeAdapter<Map<?, ?>>)
            gson.getAdapter(
                TypeToken.get(
                    betterToken.getSupertype(Map.class).getSubtype(LinkedHashMap.class).getType()));
    return new TypeAdapter<T>() {
      @Override
      public void write(JsonWriter out, T value) throws IOException {
        Map<?, ?> dict = Maps.newLinkedHashMap((Dict<?, ?>) value);
        dictAdapter.write(out, dict);
      }

      @Override
      public T read(JsonReader in) throws IOException {
        Map<?, ?> dict = dictAdapter.read(in);
        return (T) Dict.copyOf(Mutability.IMMUTABLE, dict);
      }
    };
  }
}