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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;

/**
 * Creates Gson type adapters for parameterized types by using a delegate parameterized type that
 * already has a registered type adapter factory.
 */
public final class DelegateTypeAdapterFactory<I, R extends I, D extends I>
    implements TypeAdapterFactory {
  private final Class<R> rawType;
  private final Class<I> intermediateToDelegateType;
  private final Class<D> delegateType;
  private final Function<R, D> rawToDelegate;
  private final Function<D, R> delegateToRaw;

  private DelegateTypeAdapterFactory(
      Class<R> rawType,
      Class<I> intermediateToDelegateType,
      Class<D> delegateType,
      Function<R, D> rawToDelegate,
      Function<D, R> delegateToRaw) {
    this.rawType = rawType;
    this.intermediateToDelegateType = intermediateToDelegateType;
    this.delegateType = delegateType;
    this.rawToDelegate = rawToDelegate;
    this.delegateToRaw = delegateToRaw;
  }

  public static final TypeAdapterFactory IMMUTABLE_MAP =
      new DelegateTypeAdapterFactory<>(
          ImmutableMap.class,
          Map.class,
          LinkedHashMap.class,
          raw -> new LinkedHashMap<>((Map<?, ?>) raw),
          delegate -> ImmutableMap.copyOf((Map<?, ?>) delegate));

  public static final TypeAdapterFactory IMMUTABLE_BIMAP =
      new DelegateTypeAdapterFactory<>(
          ImmutableBiMap.class,
          Map.class,
          LinkedHashMap.class,
          raw -> new LinkedHashMap<>((Map<?, ?>) raw),
          delegate -> ImmutableBiMap.copyOf((Map<?, ?>) delegate));

  public static final TypeAdapterFactory DICT =
      new DelegateTypeAdapterFactory<>(
          Dict.class,
          Map.class,
          LinkedHashMap.class,
          raw -> new LinkedHashMap<>((Map<?, ?>) raw),
          delegate -> Dict.immutableCopyOf((Map<?, ?>) delegate));

  public static final TypeAdapterFactory IMMUTABLE_LIST =
      new DelegateTypeAdapterFactory<>(
          ImmutableList.class,
          List.class,
          ArrayList.class,
          raw -> new ArrayList<>((List<?>) raw),
          delegate -> ImmutableList.copyOf((List<?>) delegate));

  public static final TypeAdapterFactory IMMUTABLE_SET =
      new DelegateTypeAdapterFactory<>(
          ImmutableSet.class,
          Set.class,
          LinkedHashSet.class,
          raw -> new LinkedHashSet<>((Set<?>) raw),
          delegate -> ImmutableSet.copyOf((Set<?>) delegate));

  @SuppressWarnings("unchecked")
  @Override
  @Nullable
  public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> typeToken) {
    Type type = typeToken.getType();
    if (typeToken.getRawType() != rawType || !(type instanceof ParameterizedType)) {
      return null;
    }

    com.google.common.reflect.TypeToken<?> betterToken =
        com.google.common.reflect.TypeToken.of(typeToken.getType());
    final TypeAdapter<Object> delegateAdapter =
        (TypeAdapter<Object>)
            gson.getAdapter(
                TypeToken.get(
                    betterToken
                        .getSupertype((Class<Object>) intermediateToDelegateType)
                        .getSubtype(delegateType)
                        .getType()));
    return new TypeAdapter<T>() {
      @Override
      public void write(JsonWriter out, T value) throws IOException {
        delegateAdapter.write(out, rawToDelegate.apply((R) value));
      }

      @Override
      public T read(JsonReader in) throws IOException {
        return (T) delegateToRaw.apply((D) delegateAdapter.read(in));
      }
    };
  }
}
