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

import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.common.flogger.GoogleLogger;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

/** A codec that serializes arbitrary types. */
public final class DynamicCodec implements ObjectCodec<Object> {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Class<?> type;
  private final TypeAndOffset[] fields;

  public DynamicCodec(Class<?> type) {
    this.type = type;
    this.fields = getOffsets(type);
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
    for (TypeAndOffset fieldInfo : fields) {
      serializeField(context, codedOut, obj, fieldInfo.type, fieldInfo.offset);
    }
  }

  /**
   * Serializes a field.
   *
   * @param obj the object containing the field to serialize. Can be an array or plain object.
   * @param type class of the field to serialize
   * @param offset unsafe offset into obj where the field will be found
   */
  @SuppressWarnings("LogAndThrow") // Want the full stack trace of serialization attempts.
  private void serializeField(
      SerializationContext context,
      CodedOutputStream codedOut,
      Object obj,
      Class<?> type,
      long offset)
      throws SerializationException, IOException {
    if (type.isPrimitive()) {
      if (type.equals(boolean.class)) {
        codedOut.writeBoolNoTag(unsafe().getBoolean(obj, offset));
      } else if (type.equals(byte.class)) {
        codedOut.writeRawByte(unsafe().getByte(obj, offset));
      } else if (type.equals(short.class)) {
        ByteBuffer buffer = ByteBuffer.allocate(2).putShort(unsafe().getShort(obj, offset));
        codedOut.writeRawBytes(buffer);
      } else if (type.equals(char.class)) {
        ByteBuffer buffer = ByteBuffer.allocate(2).putChar(unsafe().getChar(obj, offset));
        codedOut.writeRawBytes(buffer);
      } else if (type.equals(int.class)) {
        codedOut.writeInt32NoTag(unsafe().getInt(obj, offset));
      } else if (type.equals(long.class)) {
        codedOut.writeInt64NoTag(unsafe().getLong(obj, offset));
      } else if (type.equals(float.class)) {
        codedOut.writeFloatNoTag(unsafe().getFloat(obj, offset));
      } else if (type.equals(double.class)) {
        codedOut.writeDoubleNoTag(unsafe().getDouble(obj, offset));
      } else if (type.equals(void.class)) {
        // Does nothing for void type.
      } else {
        throw new UnsupportedOperationException("Unknown primitive type: " + type);
      }
    } else if (type.isArray()) {
      Object arr = unsafe().getObject(obj, offset);
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
      int base = unsafe().arrayBaseOffset(type);
      int scale = unsafe().arrayIndexScale(type);
      if (scale == 0) {
        throw new SerializationException("Failed to get index scale for type: " + type);
      }
      for (int i = 0; i < length; ++i) {
        // Serializes the ith array field directly from array memory.
        serializeField(context, codedOut, arr, type.getComponentType(), base + scale * i);
      }
    } else {
      try {
        context.serialize(unsafe().getObject(obj, offset), codedOut);
      } catch (SerializationException e) {
        logger.atSevere().withCause(e).log(
            "Unserializable object and superclass: %s %s", obj, obj.getClass().getSuperclass());
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
      instance = unsafe().allocateInstance(type);
    } catch (ReflectiveOperationException e) {
      throw new SerializationException("Could not instantiate object of type: " + type, e);
    }
    context.registerInitialValue(instance);
    for (TypeAndOffset fieldInfo : fields) {
      deserializeField(context, codedIn, instance, fieldInfo.type, fieldInfo.offset);
    }
    return instance;
  }

  /**
   * Deserializes a field directly into the supplied object.
   *
   * @param obj the object containing the field to deserialize. Can be an array or a plain object.
   * @param fieldType class of the field to deserialize
   * @param offset unsafe offset into obj where the field should be written
   */
  @SuppressWarnings("LogAndThrow") // Want the full stack trace of deserialization attempts.
  private void deserializeField(
      DeserializationContext context,
      CodedInputStream codedIn,
      Object obj,
      Class<?> fieldType,
      long offset)
      throws SerializationException, IOException {
    if (fieldType.isPrimitive()) {
      if (fieldType.equals(boolean.class)) {
        unsafe().putBoolean(obj, offset, codedIn.readBool());
      } else if (fieldType.equals(byte.class)) {
        unsafe().putByte(obj, offset, codedIn.readRawByte());
      } else if (fieldType.equals(short.class)) {
        ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
        unsafe().putShort(obj, offset, buffer.getShort(0));
      } else if (fieldType.equals(char.class)) {
        ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
        unsafe().putChar(obj, offset, buffer.getChar(0));
      } else if (fieldType.equals(int.class)) {
        unsafe().putInt(obj, offset, codedIn.readInt32());
      } else if (fieldType.equals(long.class)) {
        unsafe().putLong(obj, offset, codedIn.readInt64());
      } else if (fieldType.equals(float.class)) {
        unsafe().putFloat(obj, offset, codedIn.readFloat());
      } else if (fieldType.equals(double.class)) {
        unsafe().putDouble(obj, offset, codedIn.readDouble());
      } else if (fieldType.equals(void.class)) {
        // Does nothing for void type.
      } else {
        throw new UnsupportedOperationException(
            "Unknown primitive field type " + fieldType + " for " + type);
      }
    } else if (fieldType.isArray()) {
      if (fieldType.getComponentType().equals(byte.class)) {
        if (codedIn.readBool()) {
          unsafe().putObject(obj, offset, codedIn.readByteArray());
        } // Otherwise it was null.
        return;
      }
      int length = codedIn.readInt32();
      if (length < 0) {
        return;
      }
      Object arr = Array.newInstance(fieldType.getComponentType(), length);
      unsafe().putObject(obj, offset, arr);
      int base = unsafe().arrayBaseOffset(fieldType);
      int scale = unsafe().arrayIndexScale(fieldType);
      if (scale == 0) {
        throw new SerializationException(
            "Failed to get index scale for field type " + fieldType + " for " + type);
      }
      for (int i = 0; i < length; ++i) {
        // Deserializes type directly into array memory.
        deserializeField(context, codedIn, arr, fieldType.getComponentType(), base + scale * i);
      }
    } else {
      Object fieldValue;
      try {
        fieldValue = context.deserialize(codedIn);
      } catch (SerializationException e) {
        logger.atSevere().withCause(e).log(
            "Failed to deserialize object with superclass: %s %s",
            obj, obj.getClass().getSuperclass());
        e.addTrail(this.type);
        throw e;
      }
      if (fieldValue == null) {
        return;
      }
      if (!fieldType.isInstance(fieldValue)) {
        throw new SerializationException(
            "Field "
                + fieldValue
                + " was not instance of "
                + fieldType
                + " (was "
                + fieldValue.getClass()
                + ") for "
                + type);
      }
      unsafe().putObject(obj, offset, fieldValue);
    }
  }

  private static <T> TypeAndOffset[] getOffsets(Class<T> type) {
    // NB: it's tempting to try to simplify this by ordering by offset, but it looks like offsets
    // are not guaranteed to be stable, which is needed for deterministic serialization.
    TreeMap<Field, Long> fields = new TreeMap<>(new FieldComparator());
    for (Class<? super T> next = type; next != null; next = next.getSuperclass()) {
      for (Field field : next.getDeclaredFields()) {
        if ((field.getModifiers() & (Modifier.STATIC | Modifier.TRANSIENT)) != 0) {
          continue; // Skips static or transient fields.
        }
        fields.put(field, unsafe().objectFieldOffset(field));
      }
    }
    // Converts to an array to avoid iterator allocation when iterating.
    TypeAndOffset[] fieldsArr = new TypeAndOffset[fields.size()];
    int i = 0;
    for (Map.Entry<Field, Long> entry : fields.entrySet()) {
      fieldsArr[i++] = new TypeAndOffset(entry.getKey().getType(), entry.getValue());
    }
    return fieldsArr;
  }

  private static final class TypeAndOffset {
    private final Class<?> type;
    private final long offset;

    private TypeAndOffset(Class<?> type, long offset) {
      this.type = type;
      this.offset = offset;
    }
  }

  private static final class FieldComparator implements Comparator<Field> {
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
