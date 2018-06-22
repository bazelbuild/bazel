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

import com.google.devtools.build.lib.unsafe.UnsafeProvider;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.EnumMap;
import java.util.Map;
import sun.misc.Unsafe;

/**
 * Serialize {@link EnumMap}. Subclasses of {@link EnumMap} will crash at runtime because currently
 * there are no "benign" subclasses of {@link EnumMap} in the Bazel codebase that can be used where
 * an {@link EnumMap} was expected.
 */
class EnumMapCodec<E extends Enum<E>, V> implements ObjectCodec<EnumMap<E, V>> {
  // Only needed for empty EnumMaps where we can't figure out the Enum class from a sample element.
  private final long classTypeOffset;

  EnumMapCodec() {
    try {
      classTypeOffset =
          UnsafeProvider.getInstance().objectFieldOffset(EnumMap.class.getDeclaredField("keyType"));
    } catch (NoSuchFieldException e) {
      throw new IllegalStateException("Couldn't get keyType field fron EnumMap", e);
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public Class<EnumMap<E, V>> getEncodedClass() {
    return (Class<EnumMap<E, V>>) (Class<?>) EnumMap.class;
  }

  @Override
  public void serialize(SerializationContext context, EnumMap<E, V> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!obj.getClass().equals(EnumMap.class)) {
      throw new SerializationException(
          "Cannot serialize subclasses of EnumMap: " + obj.getClass() + " (" + obj + ")");
    }
    codedOut.writeInt32NoTag(obj.size());
    if (obj.isEmpty()) {
      // Do gross hack to get key type of map, since we have no concrete element to examine.
      Unsafe unsafe = UnsafeProvider.getInstance();
      context.serialize(unsafe.getObject(obj, classTypeOffset), codedOut);
      return;
    }
    context.serialize(obj.keySet().iterator().next().getDeclaringClass(), codedOut);
    for (Map.Entry<E, V> entry : obj.entrySet()) {
      codedOut.writeInt32NoTag(entry.getKey().ordinal());
      context.serialize(entry.getValue(), codedOut);
    }
  }

  @Override
  public EnumMap<E, V> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    Class<E> clazz = context.deserialize(codedIn);
    EnumMap<E, V> result = new EnumMap<>(clazz);
    E[] enums = clazz.getEnumConstants();
    for (int i = 0; i < size; i++) {
      int ordinal = codedIn.readInt32();
      V val = context.deserialize(codedIn);
      result.put(enums[ordinal], val);
    }
    return result;
  }
}
