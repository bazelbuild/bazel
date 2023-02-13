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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.gson.Gson;
import com.google.gson.TypeAdapter;
import com.google.gson.TypeAdapterFactory;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import javax.inject.Inject;

/**
 * Serializes and deserializes {@link ImmutableList} instances using an {@link ArrayList} as an
 * intermediary.
 */
final public class ImmutableListTypeAdatperFactory implements TypeAdapterFactory {

  @Inject
  public ImmutableListTypeAdatperFactory() {}

  @SuppressWarnings("unchecked")
  @Override
  public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> typeToken) {
    Type type = typeToken.getType();
    if (typeToken.getRawType() != ImmutableList.class || !(type instanceof ParameterizedType)) {
      return null;
    }

    com.google.common.reflect.TypeToken<ImmutableList<?>> betterToken =
        (com.google.common.reflect.TypeToken<ImmutableList<?>>)
            com.google.common.reflect.TypeToken.of(typeToken.getType());
    final TypeAdapter<ArrayList<?>> arrayListAdapter =
        (TypeAdapter<ArrayList<?>>)
            gson.getAdapter(
                TypeToken.get(
                    betterToken.getSupertype(List.class).getSubtype(ArrayList.class).getType()));
    return new TypeAdapter<T>() {
      @Override
      public void write(JsonWriter out, T value) throws IOException {
        ArrayList<?> arrayList = Lists.newArrayList((List<?>) value);
        arrayListAdapter.write(out, arrayList);
      }

      @Override
      public T read(JsonReader in) throws IOException {
        ArrayList<?> arrayList = arrayListAdapter.read(in);
        return (T) ImmutableList.copyOf(arrayList);
      }
    };
  }
}