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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.devtools.build.lib.skyframe.serialization.CodecHelpers.readChar;
import static com.google.devtools.build.lib.skyframe.serialization.CodecHelpers.readShort;
import static com.google.devtools.build.lib.skyframe.serialization.CodecHelpers.writeChar;
import static com.google.devtools.build.lib.skyframe.serialization.CodecHelpers.writeShort;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import sun.misc.Unsafe;

/**
 * Stateless class that encodes and decodes arrays that may be multi-dimensional.
 *
 * <p>Clients should obtain instances using {@link #forType}.
 */
abstract class ArrayProcessor {
  /**
   * Serializes an array.
   *
   * @param type the type of the array. Can be a multidimensional array type.
   * @param arr the array instance to serialize.
   */
  abstract void serialize(
      SerializationContext context, CodedOutputStream codedOut, Class<?> type, Object arr)
      throws IOException, SerializationException;

  /**
   * Deserializes an array into {@code obj} at {@code offset}.
   *
   * <p>A {@code (obj, offset)} tuple specifies where to write the array. Note that this
   * representation works whether {@code obj} is an array or non-array object.
   *
   * @param type the type of the array.
   * @param obj the object to contain the array, that could be an array itself.
   * @param offset offset within obj to write the deserialized value.
   */
  abstract void deserialize(
      AsyncDeserializationContext context,
      CodedInputStream codedIn,
      Class<?> type,
      Object obj,
      long offset)
      throws IOException, SerializationException;

  static ArrayProcessor forType(Class<?> type) {
    checkArgument(type.isArray(), "%s is not an array", type);
    ArrayProcessor processor;
    Class<?> baseType = resolveBaseArrayType(type);
    if (baseType.isPrimitive()) {
      if (baseType.equals(boolean.class)) {
        processor = BOOLEAN_ARRAY_PROCESSOR;
      } else if (baseType.equals(byte.class)) {
        processor = BYTE_ARRAY_PROCESSOR;
      } else if (baseType.equals(short.class)) {
        processor = SHORT_ARRAY_PROCESSOR;
      } else if (baseType.equals(char.class)) {
        processor = CHAR_ARRAY_PROCESSOR;
      } else if (baseType.equals(int.class)) {
        processor = INT_ARRAY_PROCESSOR;
      } else if (baseType.equals(long.class)) {
        processor = LONG_ARRAY_PROCESSOR;
      } else if (baseType.equals(float.class)) {
        processor = FLOAT_ARRAY_PROCESSOR;
      } else if (baseType.equals(double.class)) {
        processor = DOUBLE_ARRAY_PROCESSOR;
      } else {
        throw new UnsupportedOperationException(
            "Unexpected primitive field type " + baseType + " for " + type);
      }
    } else {
      processor = OBJECT_ARRAY_PROCESSOR;
    }
    return processor;
  }

  private static Class<?> resolveBaseArrayType(Class<?> arrayType) {
    Class<?> componentType = arrayType.getComponentType();
    if (componentType.isArray()) {
      return resolveBaseArrayType(componentType);
    }
    return componentType;
  }

  /**
   * Implements common functionality for handling multi-dimensional arrays.
   *
   * <p>Subclasses handle the base arrays by implementing {@link #serializeArrayData} and {@link
   * #deserializeArrayData}.
   */
  private abstract static class AbstractArrayProcessor extends ArrayProcessor {
    @Override
    final void serialize(
        SerializationContext context, CodedOutputStream codedOut, Class<?> type, Object arr)
        throws IOException, SerializationException {
      // The first field is a tag indicating either null or the size of the array to come.
      //   * 0 for null; or
      //   * length + 1 otherwise.
      // -1 could make sense for nulls, but nulls are fairly common and -1 has a 10 byte signed
      // integer representation. Offsetting the length like this could overflow for a length of
      // Integer.MAX_INT, but it's impossible to serialize an array of that length anyway.
      //
      // Immediately below, a tag is written for null. When non-null, the tag depends on the length
      // of the array, which is easier to obtain after the array has been cast to its array type. So
      // if it's not a nested array, the tag is written by subclasses.
      if (arr == null) {
        codedOut.writeInt32NoTag(0);
        return;
      }

      Class<?> componentType = type.getComponentType();
      if (componentType.isArray()) {
        Object[] subarrays = (Object[]) arr;
        codedOut.writeInt32NoTag(subarrays.length + 1);
        for (Object subarray : subarrays) {
          serialize(context, codedOut, componentType, subarray);
        }
        return;
      }

      serializeArrayData(context, codedOut, arr);
    }

    abstract void serializeArrayData(
        SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
        throws IOException, SerializationException;

    @Override
    final void deserialize(
        AsyncDeserializationContext context,
        CodedInputStream codedIn,
        Class<?> type,
        Object obj,
        long offset)
        throws IOException, SerializationException {
      int length = codedIn.readInt32();
      if (length == 0) {
        return; // It was null.
      }
      length--; // Shifts the length back. It was shifted to allow 0 to be used for null.

      Class<?> componentType = type.getComponentType();
      Object arr = Array.newInstance(componentType, length);
      unsafe().putObject(obj, offset, arr);

      if (length == 0) {
        return; // Empty array.
      }

      // It's a non-empty array if this is reached.
      if (componentType.isArray()) {
        for (int i = 0; i < length; ++i) {
          deserialize(
              context, codedIn, componentType, arr, OBJECT_ARR_OFFSET + OBJECT_ARR_SCALE * i);
        }
        return;
      }

      deserializeArrayData(context, codedIn, arr, length);
    }

    abstract void deserializeArrayData(
        AsyncDeserializationContext context,
        CodedInputStream codedIn,
        Object untypedArr,
        int length)
        throws IOException, SerializationException;
  }

  private static final ArrayProcessor BOOLEAN_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          boolean[] values = (boolean[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (boolean value : values) {
            codedOut.writeBoolNoTag(value);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException {
          boolean[] values = (boolean[]) untypedArr;
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readBool();
          }
        }
      };

  private static final ArrayProcessor BYTE_ARRAY_PROCESSOR =
      new ArrayProcessor() {
        // Special case uses `CodedOutputStream.writeRawBytes` for byte array. Has a different
        // control flow to allow `CodedInputStream.readRawBytes` to allocate the deserialized array.

        @Override
        public void serialize(
            SerializationContext context, CodedOutputStream codedOut, Class<?> type, Object arr)
            throws IOException, SerializationException {
          // Writes the usual offset by 1 tag for an array, see AbstractArrayProcessor.serialize.
          if (arr == null) {
            codedOut.writeInt32NoTag(0);
            return;
          }

          Class<?> componentType = type.getComponentType();
          if (componentType.isArray()) {
            // It's a multidimensional array. Serializes each component array recursively.
            Object[] subarrays = (Object[]) arr;
            codedOut.writeInt32NoTag(subarrays.length + 1);
            for (Object subArr : subarrays) {
              serialize(context, codedOut, componentType, subArr);
            }
            return;
          }

          byte[] values = (byte[]) arr;
          int length = values.length;
          codedOut.writeInt32NoTag(length + 1);
          if (length == 0) {
            return;
          }
          codedOut.writeRawBytes(values);
        }

        @Override
        public void deserialize(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Class<?> type,
            Object obj,
            long offset)
            throws IOException, SerializationException {
          int length = codedIn.readInt32();
          if (length == 0) {
            return; // It was null.
          }
          length--; // Shifts the length back. It was shifted to allow 0 to be used for null.

          Class<?> componentType = type.getComponentType();
          if (componentType.isArray()) {
            Object arr = Array.newInstance(componentType, length);
            unsafe().putObject(obj, offset, arr);
            for (int i = 0; i < length; ++i) {
              deserialize(
                  context, codedIn, componentType, arr, OBJECT_ARR_OFFSET + OBJECT_ARR_SCALE * i);
            }
            return;
          }

          unsafe().putObject(obj, offset, codedIn.readRawBytes(length));
        }
      };

  private static final ArrayProcessor SHORT_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          short[] values = (short[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (short value : values) {
            writeShort(codedOut, value);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException {
          short[] values = (short[]) untypedArr;
          for (int i = 0; i < length; ++i) {
            values[i] = readShort(codedIn);
          }
        }
      };

  private static final ArrayProcessor CHAR_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          char[] values = (char[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (char value : values) {
            writeChar(codedOut, value);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException {
          char[] values = (char[]) untypedArr;
          for (int i = 0; i < length; ++i) {
            values[i] = readChar(codedIn);
          }
        }
      };

  private static final ArrayProcessor INT_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          int[] values = (int[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (int value : values) {
            codedOut.writeInt32NoTag(value);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException {
          int[] values = (int[]) untypedArr;
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readInt32();
          }
        }
      };

  private static final ArrayProcessor LONG_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          long[] values = (long[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (long value : values) {
            codedOut.writeInt64NoTag(value);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException {
          long[] values = (long[]) untypedArr;
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readInt64();
          }
        }
      };

  private static final ArrayProcessor FLOAT_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          float[] values = (float[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (float value : values) {
            codedOut.writeFloatNoTag(value);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException {
          float[] values = (float[]) untypedArr;
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readFloat();
          }
        }
      };

  private static final ArrayProcessor DOUBLE_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          double[] values = (double[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (double value : values) {
            codedOut.writeDoubleNoTag(value);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException {
          double[] values = (double[]) untypedArr;
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readDouble();
          }
        }
      };

  private static final ArrayProcessor OBJECT_ARRAY_PROCESSOR =
      new AbstractArrayProcessor() {
        @Override
        void serializeArrayData(
            SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
            throws IOException, SerializationException {
          Object[] values = (Object[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (Object obj : values) {
            context.serialize(obj, codedOut);
          }
        }

        @Override
        public void deserializeArrayData(
            AsyncDeserializationContext context,
            CodedInputStream codedIn,
            Object untypedArr,
            int length)
            throws IOException, SerializationException {
          for (int i = 0; i < length; ++i) {
            context.deserialize(codedIn, untypedArr, OBJECT_ARR_OFFSET + OBJECT_ARR_SCALE * i);
          }
        }
      };

  private static final int OBJECT_ARR_OFFSET = Unsafe.ARRAY_OBJECT_BASE_OFFSET;
  private static final int OBJECT_ARR_SCALE = Unsafe.ARRAY_OBJECT_INDEX_SCALE;
}
