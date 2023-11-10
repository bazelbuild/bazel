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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext.FieldSetter;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

/** A codec that serializes arbitrary types. */
public final class DynamicCodec implements ObjectCodec<Object> {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Class<?> type;
  private final FieldHandler[] handlers;

  public DynamicCodec(Class<?> type) {
    this.type = type;
    this.handlers = getFieldHandlers(type);
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
  @SuppressWarnings("LogAndThrow") // Want the full stack trace.
  public void serialize(SerializationContext context, Object obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    for (FieldHandler handler : handlers) {
      try {
        handler.serialize(context, codedOut, obj);
      } catch (SerializationException e) {
        logger.atSevere().withCause(e).log(
            "Unserializable object and superclass: %s %s", obj, obj.getClass().getSuperclass());
        e.addTrail(type);
        throw e;
      }
    }
  }

  @Override
  @SuppressWarnings("LogAndThrow") // Want the full stack trace.
  public Object deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Object instance;
    try {
      instance = unsafe().allocateInstance(type);
    } catch (ReflectiveOperationException e) {
      throw new SerializationException("Could not instantiate object of type: " + type, e);
    }
    context.registerInitialValue(instance);

    for (FieldHandler handler : handlers) {
      try {
        handler.deserialize(context, codedIn, instance);
      } catch (SerializationException e) {
        logger.atSevere().withCause(e).log(
            "Failed to deserialize object with superclass: %s %s",
            instance, instance.getClass().getSuperclass());
        e.addTrail(type);
        throw e;
      }
    }
    return instance;
  }

  /** Handles serialization of a field. */
  private interface FieldHandler {
    void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws SerializationException, IOException;

    void deserialize(AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws SerializationException, IOException;
  }

  private static final class BooleanHandler implements FieldHandler {
    private final long offset;

    private BooleanHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      codedOut.writeBoolNoTag(unsafe().getBoolean(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putBoolean(obj, offset, codedIn.readBool());
    }
  }

  private static final class ByteHandler implements FieldHandler {
    private final long offset;

    private ByteHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      codedOut.writeRawByte(unsafe().getByte(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putByte(obj, offset, codedIn.readRawByte());
    }
  }

  private static void writeShort(CodedOutputStream codedOut, short value) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).putShort(value);
    codedOut.writeRawBytes(buffer);
  }

  private static short readShort(CodedInputStream codedIn) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
    return buffer.getShort(0);
  }

  private static final class ShortHandler implements FieldHandler {
    private final long offset;

    private ShortHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      writeShort(codedOut, unsafe().getShort(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putShort(obj, offset, readShort(codedIn));
    }
  }

  private static void writeChar(CodedOutputStream codedOut, char value) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).putChar(value);
    codedOut.writeRawBytes(buffer);
  }

  private static char readChar(CodedInputStream codedIn) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
    return buffer.getChar(0);
  }

  private static final class CharHandler implements FieldHandler {
    private final long offset;

    private CharHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      writeChar(codedOut, unsafe().getChar(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putChar(obj, offset, readChar(codedIn));
    }
  }

  private static final class IntHandler implements FieldHandler {
    private final long offset;

    private IntHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      codedOut.writeInt32NoTag(unsafe().getInt(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putInt(obj, offset, codedIn.readInt32());
    }
  }

  private static final class LongHandler implements FieldHandler {
    private final long offset;

    private LongHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      codedOut.writeInt64NoTag(unsafe().getLong(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putLong(obj, offset, codedIn.readInt64());
    }
  }

  private static final class FloatHandler implements FieldHandler {
    private final long offset;

    private FloatHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      codedOut.writeFloatNoTag(unsafe().getFloat(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putFloat(obj, offset, codedIn.readFloat());
    }
  }

  private static final class DoubleHandler implements FieldHandler {
    private final long offset;

    private DoubleHandler(long offset) {
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException {
      codedOut.writeDoubleNoTag(unsafe().getDouble(obj, offset));
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException {
      unsafe().putDouble(obj, offset, codedIn.readDouble());
    }
  }

  private static final class ObjectHandler implements FieldHandler, FieldSetter {
    private final Class<?> type;
    private final long offset;

    private ObjectHandler(Class<?> type, long offset) {
      this.type = type;
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException, SerializationException {
      context.serialize(unsafe().getObject(obj, offset), codedOut);
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException, SerializationException {
      context.deserialize(codedIn, obj, (FieldSetter) this);
    }

    @Override
    public void set(Object target, Object fieldValue) throws SerializationException {
      if (!type.isInstance(fieldValue)) {
        throw new SerializationException(
            "Field "
                + fieldValue
                + " was not instance of "
                + type
                + " (was "
                + fieldValue.getClass()
                + ")");
      }
      unsafe().putObject(target, offset, fieldValue);
    }
  }

  private static final class ArrayHandler implements FieldHandler {
    private final Class<?> type;
    private final long offset;

    private ArrayHandler(Class<?> type, long offset) {
      checkArgument(type.isArray(), type);
      this.type = type;
      this.offset = offset;
    }

    @Override
    public void serialize(SerializationContext context, CodedOutputStream codedOut, Object obj)
        throws IOException, SerializationException {
      serializeArray(context, codedOut, type, obj, offset);
    }

    /**
     * Serializes an array.
     *
     * <p>A {@code (obj, offset)} tuple specifies the array. Note that this representation works
     * regardless of whether the containing {@code obj} is an array or non-array object. This method
     * is used in both cases.
     *
     * @param type the type of the array.
     * @param obj the object containing the array. {@code obj} could be an array also.
     * @param offset offset within obj of the array.
     */
    private static void serializeArray(
        SerializationContext context,
        CodedOutputStream codedOut,
        Class<?> type,
        Object obj,
        long offset)
        throws IOException, SerializationException {
      Object arr = unsafe().getObject(obj, offset);

      Class<?> componentType = type.getComponentType();
      if (componentType.equals(byte.class)) {
        // Special case uses `CodedOutputStream.writeByteArrayNoTag` for byte array. Uses a
        // different convention for tagging arrays to avoid writing length twice.
        if (arr == null) {
          codedOut.writeBoolNoTag(false);
        } else {
          codedOut.writeBoolNoTag(true);
          codedOut.writeByteArrayNoTag((byte[]) arr);
        }
        return;
      }

      // Writes a tag indicating the size of the array to come.
      //   * 0 for null; or
      //   * length + 1 otherwise.
      // -1 could make sense for nulls, but nulls are fairly common and -1 has a 10 byte signed
      // integer representation.
      if (arr == null) {
        codedOut.writeInt32NoTag(0);
        return;
      }

      int length = Array.getLength(arr);
      // This could overflow for a length of Integer.MAX_INT, but it's impossible to serialize an
      // array of that length anyway.
      codedOut.writeInt32NoTag(length + 1);
      if (length == 0) {
        return;
      }

      // It's a non-empty array if this is reached.
      int base = unsafe().arrayBaseOffset(type);
      int scale = unsafe().arrayIndexScale(type);
      if (scale == 0) {
        throw new SerializationException("Failed to get index scale for type: " + type);
      }

      if (componentType.equals(boolean.class)) {
        // It's possible to use BitSet here to better pack the Booleans on the wire, but if the
        // array were long enough to make this efficient, the underlying code would have used a
        // different representation anyway.
        for (int i = 0; i < length; ++i) {
          codedOut.writeBoolNoTag(unsafe().getBoolean(arr, base + scale * i));
        }
      } else if (componentType.equals(short.class)) {
        for (int i = 0; i < length; ++i) {
          writeShort(codedOut, unsafe().getShort(arr, base + scale * i));
        }
      } else if (componentType.equals(char.class)) {
        for (int i = 0; i < length; ++i) {
          writeChar(codedOut, unsafe().getChar(arr, base + scale * i));
        }
      } else if (componentType.equals(int.class)) {
        for (int i = 0; i < length; ++i) {
          codedOut.writeInt32NoTag(unsafe().getInt(arr, base + scale * i));
        }
      } else if (componentType.equals(long.class)) {
        for (int i = 0; i < length; ++i) {
          codedOut.writeInt64NoTag(unsafe().getLong(arr, base + scale * i));
        }
      } else if (componentType.equals(float.class)) {
        for (int i = 0; i < length; ++i) {
          codedOut.writeFloatNoTag(unsafe().getFloat(arr, base + scale * i));
        }
      } else if (componentType.equals(double.class)) {
        for (int i = 0; i < length; ++i) {
          codedOut.writeDoubleNoTag(unsafe().getDouble(arr, base + scale * i));
        }
      } else if (componentType.isArray()) {
        // It's a multidimensional array. Serializes each component array recursively.
        for (int i = 0; i < length; ++i) {
          serializeArray(context, codedOut, componentType, arr, base + scale * i);
        }
      } else {
        // Otherwise, the components have some arbitrary object type.
        for (int i = 0; i < length; ++i) {
          context.serialize(unsafe().getObject(arr, base + scale * i), codedOut);
        }
      }
    }

    @Override
    public void deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Object obj)
        throws IOException, SerializationException {
      deserializeArray(context, codedIn, type, obj, offset);
    }

    private static void deserializeArray(
        AsyncDeserializationContext context,
        CodedInputStream codedIn,
        Class<?> type,
        Object obj,
        long offset)
        throws IOException, SerializationException {
      Class<?> componentType = type.getComponentType();

      if (componentType.equals(byte.class)) {
        if (codedIn.readBool()) {
          unsafe().putObject(obj, offset, codedIn.readByteArray());
        } // Otherwise, it was null.
        return;
      }

      int length = codedIn.readInt32();
      if (length == 0) {
        return; // It was null.
      }
      length--; // Shifts the length back. They were shifted to allow 0 to be used for null.
      Object arr = Array.newInstance(componentType, length);
      unsafe().putObject(obj, offset, arr);

      if (length == 0) {
        return; // Empty array.
      }

      // It's a non-empty array if this is reached.
      int base = unsafe().arrayBaseOffset(type);
      int scale = unsafe().arrayIndexScale(type);
      if (scale == 0) {
        throw new SerializationException("Failed to get index scale for type: " + type);
      }

      if (componentType.equals(boolean.class)) {
        for (int i = 0; i < length; ++i) {
          unsafe().putBoolean(arr, base + scale * i, codedIn.readBool());
        }
      } else if (componentType.equals(short.class)) {
        for (int i = 0; i < length; ++i) {
          unsafe().putShort(arr, base + scale * i, readShort(codedIn));
        }
      } else if (componentType.equals(char.class)) {
        for (int i = 0; i < length; ++i) {
          unsafe().putChar(arr, base + scale * i, readChar(codedIn));
        }
        return;
      } else if (componentType.equals(int.class)) {
        for (int i = 0; i < length; ++i) {
          unsafe().putInt(arr, base + scale * i, codedIn.readInt32());
        }
      } else if (componentType.equals(long.class)) {
        for (int i = 0; i < length; ++i) {
          unsafe().putLong(arr, base + scale * i, codedIn.readInt64());
        }
      } else if (componentType.equals(float.class)) {
        for (int i = 0; i < length; ++i) {
          unsafe().putFloat(arr, base + scale * i, codedIn.readFloat());
        }
      } else if (componentType.equals(double.class)) {
        for (int i = 0; i < length; ++i) {
          unsafe().putDouble(arr, base + scale * i, codedIn.readDouble());
        }
      } else if (componentType.isArray()) {
        // It's a multidimensional array. Deserializes each component array recursively.
        for (int i = 0; i < length; ++i) {
          deserializeArray(context, codedIn, componentType, arr, base + scale * i);
        }
      } else {
        // Otherwise, the components have some arbitrary object type.
        for (int i = 0; i < length; ++i) {
          context.deserialize(codedIn, arr, base + scale * i);
        }
      }
    }
  }

  private static <T> FieldHandler[] getFieldHandlers(Class<T> type) {
    // NB: it's tempting to try to simplify this by ordering by offset, but it looks like offsets
    // are not guaranteed to be stable, which is needed for deterministic serialization.
    ArrayList<Field> fields = new ArrayList<>();
    for (Class<? super T> next = type; next != null; next = next.getSuperclass()) {
      for (Field field : next.getDeclaredFields()) {
        if ((field.getModifiers() & (Modifier.STATIC | Modifier.TRANSIENT)) != 0) {
          continue; // Skips static or transient fields.
        }
        fields.add(field);
      }
    }
    Collections.sort(fields, new FieldComparator());
    FieldHandler[] handlers = new FieldHandler[fields.size()];
    int i = 0;
    for (Field field : fields) {
      long offset = unsafe().objectFieldOffset(field);
      Class<?> fieldType = field.getType();
      FieldHandler handler;
      if (fieldType.isPrimitive()) {
        if (fieldType.equals(boolean.class)) {
          handler = new BooleanHandler(offset);
        } else if (fieldType.equals(byte.class)) {
          handler = new ByteHandler(offset);
        } else if (fieldType.equals(short.class)) {
          handler = new ShortHandler(offset);
        } else if (fieldType.equals(char.class)) {
          handler = new CharHandler(offset);
        } else if (fieldType.equals(int.class)) {
          handler = new IntHandler(offset);
        } else if (fieldType.equals(long.class)) {
          handler = new LongHandler(offset);
        } else if (fieldType.equals(float.class)) {
          handler = new FloatHandler(offset);
        } else if (fieldType.equals(double.class)) {
          handler = new DoubleHandler(offset);
        } else {
          throw new UnsupportedOperationException(
              "Unexpected primitive field type " + fieldType + " for " + type);
        }
      } else if (fieldType.isArray()) {
        handler = new ArrayHandler(fieldType, offset);
      } else {
        handler = new ObjectHandler(fieldType, offset);
      }
      handlers[i++] = handler;
    }
    return handlers;
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
