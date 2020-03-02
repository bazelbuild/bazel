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
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Logger;
import sun.reflect.ReflectionFactory;

/**
 * A codec that serializes arbitrary types.
 *
 * <p>TODO(shahan): replace Unsafe with VarHandle once it's available.
 */
public class DynamicCodec implements ObjectCodec<Object> {
  private static final Logger logger = Logger.getLogger(DynamicCodec.class.getName());

  private final Class<?> type;
  private final Constructor<?> constructor;
  private final TypeAndOffset[] offsets;

  public DynamicCodec(Class<?> type) throws ReflectiveOperationException {
    this.type = type;
    this.constructor = getConstructor(type);
    this.offsets = getOffsets(type);
  }

  @Override
  public Class<?> getEncodedClass() {
    return type;
  }

  @Override
  public MemoizationStrategy getStrategy() {
    return ObjectCodec.MemoizationStrategy.MEMOIZE_BEFORE;
  }

  @Override
  public void serialize(SerializationContext context, Object obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    for (int i = 0; i < offsets.length; ++i) {
      serializeField(context, codedOut, obj, offsets[i].type, offsets[i].offset);
    }
  }

  /**
   * Serializes a field.
   *
   * @param obj the object containing the field to serialize. Can be an array or plain object.
   * @param type class of the field to serialize
   * @param offset unsafe offset into obj where the field will be found
   */
  private void serializeField(
      SerializationContext context,
      CodedOutputStream codedOut,
      Object obj,
      Class<?> type,
      long offset)
      throws SerializationException, IOException {
    if (type.isPrimitive()) {
      if (type.equals(boolean.class)) {
        codedOut.writeBoolNoTag(UnsafeProvider.getInstance().getBoolean(obj, offset));
      } else if (type.equals(byte.class)) {
        codedOut.writeRawByte(UnsafeProvider.getInstance().getByte(obj, offset));
      } else if (type.equals(short.class)) {
        ByteBuffer buffer =
            ByteBuffer.allocate(2).putShort(UnsafeProvider.getInstance().getShort(obj, offset));
        codedOut.writeRawBytes(buffer);
      } else if (type.equals(char.class)) {
        ByteBuffer buffer =
            ByteBuffer.allocate(2).putChar(UnsafeProvider.getInstance().getChar(obj, offset));
        codedOut.writeRawBytes(buffer);
      } else if (type.equals(int.class)) {
        codedOut.writeInt32NoTag(UnsafeProvider.getInstance().getInt(obj, offset));
      } else if (type.equals(long.class)) {
        codedOut.writeInt64NoTag(UnsafeProvider.getInstance().getLong(obj, offset));
      } else if (type.equals(float.class)) {
        codedOut.writeFloatNoTag(UnsafeProvider.getInstance().getFloat(obj, offset));
      } else if (type.equals(double.class)) {
        codedOut.writeDoubleNoTag(UnsafeProvider.getInstance().getDouble(obj, offset));
      } else if (type.equals(void.class)) {
        // Does nothing for void type.
      } else {
        throw new UnsupportedOperationException("Unknown primitive type: " + type);
      }
    } else if (type.isArray()) {
      Object arr = UnsafeProvider.getInstance().getObject(obj, offset);
      if (type.getComponentType().equals(byte.class)) {
        if (arr == null) {
          codedOut.writeBoolNoTag(false);
        } else {
          codedOut.writeBoolNoTag(true);
          codedOut.writeByteArrayNoTag((byte[]) arr);
        }
        return;
      }
      if (arr == null) {
        codedOut.writeInt32NoTag(-1);
        return;
      }
      int length = Array.getLength(arr);
      codedOut.writeInt32NoTag(length);
      int base = UnsafeProvider.getInstance().arrayBaseOffset(type);
      int scale = UnsafeProvider.getInstance().arrayIndexScale(type);
      if (scale == 0) {
        throw new SerializationException("Failed to get index scale for type: " + type);
      }
      for (int i = 0; i < length; ++i) {
        // Serializes the ith array field directly from array memory.
        serializeField(context, codedOut, arr, type.getComponentType(), base + scale * i);
      }
    } else {
      try {
        context.serialize(UnsafeProvider.getInstance().getObject(obj, offset), codedOut);
      } catch (SerializationException e) {
        logger.severe(
            String.format(
                "Unserializable object and superclass: %s %s",
                obj, obj.getClass().getSuperclass()));
        e.addTrail(this.type);
        throw e;
      }
    }
  }

  @Override
  public Object deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Object instance;
    try {
      instance = constructor.newInstance();
    } catch (ReflectiveOperationException e) {
      throw new SerializationException("Could not instantiate object of type: " + type, e);
    }
    context.registerInitialValue(instance);
    for (int i = 0; i < offsets.length; ++i) {
      deserializeField(context, codedIn, instance, offsets[i].type, offsets[i].offset);
    }
    return instance;
  }

  /**
   * Deserializes a field directly into the supplied object.
   *
   * @param obj the object containing the field to deserialize. Can be an array or a plain object.
   * @param type class of the field to deserialize
   * @param offset unsafe offset into obj where the field should be written
   */
  private void deserializeField(
      DeserializationContext context,
      CodedInputStream codedIn,
      Object obj,
      Class<?> type,
      long offset)
      throws SerializationException, IOException {
    if (type.isPrimitive()) {
      if (type.equals(boolean.class)) {
        UnsafeProvider.getInstance().putBoolean(obj, offset, codedIn.readBool());
      } else if (type.equals(byte.class)) {
        UnsafeProvider.getInstance().putByte(obj, offset, codedIn.readRawByte());
      } else if (type.equals(short.class)) {
        ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
        UnsafeProvider.getInstance().putShort(obj, offset, buffer.getShort(0));
      } else if (type.equals(char.class)) {
        ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
        UnsafeProvider.getInstance().putChar(obj, offset, buffer.getChar(0));
      } else if (type.equals(int.class)) {
        UnsafeProvider.getInstance().putInt(obj, offset, codedIn.readInt32());
      } else if (type.equals(long.class)) {
        UnsafeProvider.getInstance().putLong(obj, offset, codedIn.readInt64());
      } else if (type.equals(float.class)) {
        UnsafeProvider.getInstance().putFloat(obj, offset, codedIn.readFloat());
      } else if (type.equals(double.class)) {
        UnsafeProvider.getInstance().putDouble(obj, offset, codedIn.readDouble());
      } else if (type.equals(void.class)) {
        // Does nothing for void type.
      } else {
        throw new UnsupportedOperationException("Unknown primitive type: " + type);
      }
    } else if (type.isArray()) {
      if (type.getComponentType().equals(byte.class)) {
        boolean isNonNull = codedIn.readBool();
        UnsafeProvider.getInstance()
            .putObject(obj, offset, isNonNull ? codedIn.readByteArray() : null);
        return;
      }
      int length = codedIn.readInt32();
      if (length < 0) {
        UnsafeProvider.getInstance().putObject(obj, offset, null);
        return;
      }
      Object arr = Array.newInstance(type.getComponentType(), length);
      UnsafeProvider.getInstance().putObject(obj, offset, arr);
      int base = UnsafeProvider.getInstance().arrayBaseOffset(type);
      int scale = UnsafeProvider.getInstance().arrayIndexScale(type);
      if (scale == 0) {
        throw new SerializationException("Failed to get index scale for type: " + type);
      }
      for (int i = 0; i < length; ++i) {
        // Deserializes type directly into array memory.
        deserializeField(context, codedIn, arr, type.getComponentType(), base + scale * i);
      }
    } else {
      UnsafeProvider.getInstance().putObject(obj, offset, context.deserialize(codedIn));
    }
  }

  private static <T> TypeAndOffset[] getOffsets(Class<T> type) {
    TreeMap<Field, Long> offsets = new TreeMap<>(new FieldComparator());
    for (Class<? super T> next = type; next != null; next = next.getSuperclass()) {
      for (Field field : next.getDeclaredFields()) {
        if ((field.getModifiers() & (Modifier.STATIC | Modifier.TRANSIENT)) != 0) {
          continue; // Skips static or transient fields.
        }
        field.setAccessible(true);
        offsets.put(field, UnsafeProvider.getInstance().objectFieldOffset(field));
      }
    }
    // Converts to an array to make it easy to avoid the use of iterators.
    TypeAndOffset[] offsetsArr = new TypeAndOffset[offsets.size()];
    int i = 0;
    for (Map.Entry<Field, Long> entry : offsets.entrySet()) {
      offsetsArr[i] = new TypeAndOffset(entry.getKey().getType(), entry.getValue());
      ++i;
    }
    return offsetsArr;
  }

  private static class TypeAndOffset {
    public final Class<?> type;
    public final long offset;

    public TypeAndOffset(Class<?> type, long offset) {
      this.type = type;
      this.offset = offset;
    }
  }

  private static Constructor<?> getConstructor(Class<?> type) throws ReflectiveOperationException {
    Constructor<?> constructor =
        ReflectionFactory.getReflectionFactory()
            .newConstructorForSerialization(type, Object.class.getDeclaredConstructor());
    constructor.setAccessible(true);
    return constructor;
  }

  private static class FieldComparator implements Comparator<Field> {

    @Override
    public int compare(Field f1, Field f2) {
      int classCompare =
          f1.getDeclaringClass().getName().compareTo(f2.getDeclaringClass().getName());
      if (classCompare != 0) {
        return classCompare;
      }
      return f1.getName().compareTo(f2.getName());
    }
  }
}
