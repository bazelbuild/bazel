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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import javax.annotation.Nullable;

/**
 * Stateless class that encodes and decodes arrays that may be multi-dimensional.
 *
 * <p>Clients should obtain instances using {@link #forType}.
 */
public interface ArrayProcessor {
  /**
   * Serializes an array.
   *
   * @param type the type of the array. Can be a multidimensional array type.
   * @param arr the array instance to serialize.
   */
  void serialize(
      SerializationContext context, CodedOutputStream codedOut, Class<?> type, Object arr)
      throws IOException, SerializationException;

  /**
   * Deserializes an array of type {@code arrayType} from {@code codedIn}.
   *
   * @return the array object. {@link Object} is the most specific common type ancestor of {@code
   *     Object[]} and {@code int[]}.
   */
  Object deserialize(
      AsyncDeserializationContext context, CodedInputStream codedIn, Class<?> arrayType)
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

  // This method should be marked private, but that's not supported in Java 8.
  static Class<?> resolveBaseArrayType(Class<?> arrayType) {
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
  abstract static class PrimitiveArrayProcessor implements ArrayProcessor {
    @Override
    public final void serialize(
        SerializationContext context, CodedOutputStream codedOut, Class<?> type, Object arr)
        throws IOException {
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

      serializeArrayData(codedOut, arr);
    }

    public abstract void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
        throws IOException;

    @Override
    public Object deserialize(
        AsyncDeserializationContext context, CodedInputStream codedIn, Class<?> arrayType)
        throws IOException, SerializationException {
      return deserialize(codedIn, arrayType);
    }

    /** Primitive arrays can be deserialized without an {@link AsyncDeserializationContext}. */
    @Nullable
    private Object deserialize(CodedInputStream codedIn, Class<?> arrayType) throws IOException {
      int length = codedIn.readInt32();
      if (length == 0) {
        return null; // It was null.
      }
      length--; // Shifts the length back. It was shifted to allow 0 to be used for null.

      Class<?> componentType = arrayType.getComponentType();
      if (!componentType.isArray()) {
        return deserializeArrayData(codedIn, length);
      }

      var arr = (Object[]) Array.newInstance(componentType, length);
      for (int i = 0; i < length; ++i) {
        arr[i] = deserialize(codedIn, componentType);
      }
      return arr;
    }

    public abstract Object deserializeArrayData(CodedInputStream codedIn, int length)
        throws IOException;
  }

  static final PrimitiveArrayProcessor BOOLEAN_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          boolean[] values = (boolean[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (boolean value : values) {
            codedOut.writeBoolNoTag(value);
          }
        }

        @Override
        public boolean[] deserializeArrayData(CodedInputStream codedIn, int length)
            throws IOException {
          boolean[] values = new boolean[length];
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readBool();
          }
          return values;
        }
      };

  static final PrimitiveArrayProcessor BYTE_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          byte[] values = (byte[]) untypedArr;
          int length = values.length;
          codedOut.writeInt32NoTag(length + 1);
          if (length > 0) {
            codedOut.writeRawBytes(values);
          }
        }

        @Override
        public byte[] deserializeArrayData(CodedInputStream codedIn, int length)
            throws IOException {
          return codedIn.readRawBytes(length);
        }
      };

  static final PrimitiveArrayProcessor SHORT_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          short[] values = (short[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (short value : values) {
            writeShort(codedOut, value);
          }
        }

        @Override
        public short[] deserializeArrayData(CodedInputStream codedIn, int length)
            throws IOException {
          short[] values = new short[length];
          for (int i = 0; i < length; ++i) {
            values[i] = readShort(codedIn);
          }
          return values;
        }
      };

  static final PrimitiveArrayProcessor CHAR_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          char[] values = (char[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (char value : values) {
            writeChar(codedOut, value);
          }
        }

        @Override
        public char[] deserializeArrayData(CodedInputStream codedIn, int length)
            throws IOException {
          char[] values = new char[length];
          for (int i = 0; i < length; ++i) {
            values[i] = readChar(codedIn);
          }
          return values;
        }
      };

  static final PrimitiveArrayProcessor INT_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          int[] values = (int[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (int value : values) {
            codedOut.writeInt32NoTag(value);
          }
        }

        @Override
        public int[] deserializeArrayData(CodedInputStream codedIn, int length) throws IOException {
          int[] values = new int[length];
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readInt32();
          }
          return values;
        }
      };

  static final PrimitiveArrayProcessor LONG_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          long[] values = (long[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (long value : values) {
            codedOut.writeInt64NoTag(value);
          }
        }

        @Override
        public long[] deserializeArrayData(CodedInputStream codedIn, int length)
            throws IOException {
          long[] values = new long[length];
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readInt64();
          }
          return values;
        }
      };

  static final PrimitiveArrayProcessor FLOAT_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          float[] values = (float[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (float value : values) {
            codedOut.writeFloatNoTag(value);
          }
        }

        @Override
        public float[] deserializeArrayData(CodedInputStream codedIn, int length)
            throws IOException {
          float[] values = new float[length];
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readFloat();
          }
          return values;
        }
      };

  static final PrimitiveArrayProcessor DOUBLE_ARRAY_PROCESSOR =
      new PrimitiveArrayProcessor() {
        @Override
        public void serializeArrayData(CodedOutputStream codedOut, Object untypedArr)
            throws IOException {
          double[] values = (double[]) untypedArr;
          codedOut.writeInt32NoTag(values.length + 1);
          for (double value : values) {
            codedOut.writeDoubleNoTag(value);
          }
        }

        @Override
        public double[] deserializeArrayData(CodedInputStream codedIn, int length)
            throws IOException {
          double[] values = new double[length];
          for (int i = 0; i < length; ++i) {
            values[i] = codedIn.readDouble();
          }
          return values;
        }
      };

  /**
   * Handles possibly nested arrays of {@code Object} or any type derived from {@code Object}.
   *
   * <p>This processor observes the nesting level of the array by reflective operations on the
   * {@code type} parameter passed into its methods. It similarly uses reflective calls to create
   * nested arrays of appropriate type and nesting level.
   *
   * <p>Finally, at the leaf level, it uses the {@code (Ser|Deser)ializationContext} to apply codecs
   * to the base array components. The {@link DeserializationContext} must be an {@link
   * AsyncDeserializationContext}.
   */
  static final ArrayProcessor OBJECT_ARRAY_PROCESSOR =
      new ArrayProcessor() {
        // Special case uses `Array.newInstance` to also create leaf-level arrays. Unlike the
        // `PrimitiveArrayProcessor`, the unnested arrays depend on serialization contexts.

        @Override
        public void serialize(
            SerializationContext context, CodedOutputStream codedOut, Class<?> type, Object arr)
            throws IOException, SerializationException {
          // Tagging works exactly the same as PrimitiveArrayProcessor.serialize: 0 for null;
          // 1 + length otherwise. See comment there for more details.
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

          serializeObjectArray(context, codedOut, arr);
        }

        @Override
        @Nullable
        public Object deserialize(
            AsyncDeserializationContext context, CodedInputStream codedIn, Class<?> arrayType)
            throws IOException, SerializationException {
          int length = codedIn.readInt32();
          if (length == 0) {
            return null; // It was null.
          }
          length--; // Shifts the length back. It was shifted to allow 0 to be used for null.

          Class<?> componentType = arrayType.getComponentType();
          var arr = (Object[]) Array.newInstance(componentType, length);

          if (length > 0) {
            if (componentType.isArray()) {
              for (int i = 0; i < length; ++i) {
                arr[i] = deserialize(context, codedIn, componentType);
              }
            } else {
              deserializeObjectArray(context, codedIn, arr, length);
            }
          }
          return arr;
        }
      };

  /** Serializes an object array using the given {@code context} to the {@code codedOut} stream. */
  static void serializeObjectArray(
      SerializationContext context, CodedOutputStream codedOut, Object untypedArr)
      throws IOException, SerializationException {
    Object[] values = (Object[]) untypedArr;
    codedOut.writeInt32NoTag(values.length + 1);
    for (Object obj : values) {
      context.serialize(obj, codedOut);
    }
  }

  /**
   * Deserializes {@code length} objects into {@code arr}.
   *
   * <p>Partially deserialized values may be visible to the caller.
   */
  @SuppressWarnings("AvoidObjectArrays") // explicit, low-level array handling
  static void deserializeObjectArray(
      AsyncDeserializationContext context, CodedInputStream codedIn, Object[] arr, int length)
      throws IOException, SerializationException {
    for (int i = 0; i < length; ++i) {
      context.deserializeArrayElement(codedIn, arr, i);
    }
  }
}
