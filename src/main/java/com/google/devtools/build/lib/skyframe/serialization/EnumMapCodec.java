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

import static com.google.devtools.build.lib.skyframe.serialization.ClassCodec.classCodec;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Serialize {@link EnumMap}. Subclasses of {@link EnumMap} will crash at runtime because currently
 * there are no "benign" subclasses of {@link EnumMap} in the Bazel codebase that can be used where
 * an {@link EnumMap} was expected.
 */
@SuppressWarnings({"rawtypes", "unchecked"})
class EnumMapCodec extends AsyncObjectCodec<EnumMap> {
  /**
   * Used to retrieve the hidden {@link EnumMap#keyType} field.
   *
   * <p>This requires {@code java.base/java.util} to be opened to this module. Note that {@link
   * MethodHandles.Lookup#unreflectVarHandle} can't be used here because it ignores {@link
   * java.lang.reflect.Field#setAccessible}.
   */
  private static final VarHandle KEY_TYPE;

  static {
    try {
      KEY_TYPE =
          MethodHandles.privateLookupIn(EnumMap.class, MethodHandles.lookup())
              .findVarHandle(EnumMap.class, "keyType", Class.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @Override
  public Class<EnumMap> getEncodedClass() {
    return EnumMap.class;
  }

  @Override
  public void serialize(SerializationContext context, EnumMap obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!obj.getClass().equals(EnumMap.class)) {
      throw new SerializationException(
          "Cannot serialize subclasses of EnumMap: " + obj.getClass() + " (" + obj + ")");
    }
    classCodec().serialize(context, (Class<?>) KEY_TYPE.get(obj), codedOut);

    codedOut.writeInt32NoTag(obj.size());
    if (obj.isEmpty()) {
      return;
    }

    for (Object next : obj.entrySet()) {
      Map.Entry entry = (Map.Entry) next;
      codedOut.writeInt32NoTag(((Enum) entry.getKey()).ordinal());
      context.serialize(entry.getValue(), codedOut);
    }
  }

  @Override
  public EnumMap deserializeAsync(AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Class clazz = classCodec().deserialize(context, codedIn);
    int size = codedIn.readInt32();
    EnumMap result = new EnumMap(clazz);
    context.registerInitialValue(result);

    MapBuffer buffer = new MapBuffer(result, size);

    Enum[] enums = (Enum[]) clazz.getEnumConstants();
    for (int i = 0; i < size; i++) {
      int ordinal = codedIn.readInt32();
      buffer.setEnum(i, enums[ordinal]);
      context.deserializeArrayElement(codedIn, buffer.values, i, /* done= */ buffer);
    }
    return result;
  }

  /** Buffers the entry elements and populates the map once all values are done. */
  private static class MapBuffer implements Runnable {
    private final EnumMap result;
    private final Enum[] enums;
    private final Object[] values;

    private final AtomicInteger remaining;

    private MapBuffer(EnumMap result, int size) {
      this.result = result;
      this.enums = new Enum[size];
      this.values = new Object[size];
      this.remaining = new AtomicInteger(size);
    }

    @Override
    public void run() {
      if (remaining.decrementAndGet() == 0) {
        for (int i = 0; i < enums.length; i++) {
          result.put(enums[i], values[i]);
        }
      }
    }

    private void setEnum(int index, Enum enumKey) {
      enums[index] = enumKey;
    }
  }
}
