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