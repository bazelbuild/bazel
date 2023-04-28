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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;

/** {@link ObjectCodec} for arrays of an arbitrary component type. */
public class ArrayCodec<T> implements ObjectCodec<T[]> {

  /** Creates a codec for arrays of the given component type. */
  public static <T> ArrayCodec<T> forComponentType(Class<T> componentType) {
    @SuppressWarnings("unchecked")
    Class<T[]> arrayType = (Class<T[]>) Array.newInstance(componentType, 0).getClass();
    return new ArrayCodec<>(componentType, arrayType);
  }

  /** Codec for {@code Object[]}. */
  static final class ObjectArrayCodec extends ArrayCodec<Object> {
    ObjectArrayCodec() {
      super(Object.class, Object[].class);
    }
  }

  private final Class<T> componentType;
  private final Class<T[]> arrayType;

  private ArrayCodec(Class<T> componentType, Class<T[]> arrayType) {
    this.componentType = componentType;
    this.arrayType = arrayType;
  }

  @Override
  public final Class<T[]> getEncodedClass() {
    return arrayType;
  }

  @Override
  public final void serialize(SerializationContext context, T[] obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.length);
    try {
      for (Object item : obj) {
        context.serialize(item, codedOut);
      }
    } catch (StackOverflowError e) {
      // TODO(janakr): figure out if we need to handle this better and handle it better if so.
      throw new SerializationException("StackOverflow serializing array", e);
    }
  }

  @Override
  public final T[] deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    @SuppressWarnings("unchecked")
    T[] result = (T[]) Array.newInstance(componentType, codedIn.readInt32());
    try {
      for (int i = 0; i < result.length; i++) {
        result[i] = context.deserialize(codedIn);
      }
    } catch (StackOverflowError e) {
      // TODO(janakr): figure out if we need to handle this better and handle it better if so.
      throw new SerializationException("StackOverflow deserializing array", e);
    }
    return result;
  }
}
