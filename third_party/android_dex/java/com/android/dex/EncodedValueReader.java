/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.dex;

import com.android.dex.util.ByteInput;

/**
 * Pull parser for encoded values.
 */
public final class EncodedValueReader {
    public static final int ENCODED_BYTE = 0x00;
    public static final int ENCODED_SHORT = 0x02;
    public static final int ENCODED_CHAR = 0x03;
    public static final int ENCODED_INT = 0x04;
    public static final int ENCODED_LONG = 0x06;
    public static final int ENCODED_FLOAT = 0x10;
    public static final int ENCODED_DOUBLE = 0x11;
    public static final int ENCODED_METHOD_TYPE = 0x15;
    public static final int ENCODED_METHOD_HANDLE = 0x16;
    public static final int ENCODED_STRING = 0x17;
    public static final int ENCODED_TYPE = 0x18;
    public static final int ENCODED_FIELD = 0x19;
    public static final int ENCODED_ENUM = 0x1b;
    public static final int ENCODED_METHOD = 0x1a;
    public static final int ENCODED_ARRAY = 0x1c;
    public static final int ENCODED_ANNOTATION = 0x1d;
    public static final int ENCODED_NULL = 0x1e;
    public static final int ENCODED_BOOLEAN = 0x1f;

    /** placeholder type if the type is not yet known */
    private static final int MUST_READ = -1;

    protected final ByteInput in;
    private int type = MUST_READ;
    private int annotationType;
    private int arg;

    public EncodedValueReader(ByteInput in) {
        this.in = in;
    }

    public EncodedValueReader(EncodedValue in) {
        this(in.asByteInput());
    }

    /**
     * Creates a new encoded value reader whose only value is the specified
     * known type. This is useful for encoded values without a type prefix,
     * such as class_def_item's encoded_array or annotation_item's
     * encoded_annotation.
     */
    public EncodedValueReader(ByteInput in, int knownType) {
        this.in = in;
        this.type = knownType;
    }

    public EncodedValueReader(EncodedValue in, int knownType) {
        this(in.asByteInput(), knownType);
    }

    /**
     * Returns the type of the next value to read.
     */
    public int peek() {
        if (type == MUST_READ) {
            int argAndType = in.readByte() & 0xff;
            type = argAndType & 0x1f;
            arg = (argAndType & 0xe0) >> 5;
        }
        return type;
    }

    /**
     * Begins reading the elements of an array, returning the array's size. The
     * caller must follow up by calling a read method for each element in the
     * array. For example, this reads a byte array: <pre>   {@code
     *   int arraySize = readArray();
     *   for (int i = 0, i < arraySize; i++) {
     *     readByte();
     *   }
     * }</pre>
     */
    public int readArray() {
        checkType(ENCODED_ARRAY);
        type = MUST_READ;
        return Leb128.readUnsignedLeb128(in);
    }

    /**
     * Begins reading the fields of an annotation, returning the number of
     * fields. The caller must follow up by making alternating calls to {@link
     * #readAnnotationName()} and another read method. For example, this reads
     * an annotation whose fields are all bytes: <pre>   {@code
     *   int fieldCount = readAnnotation();
     *   int annotationType = getAnnotationType();
     *   for (int i = 0; i < fieldCount; i++) {
     *       readAnnotationName();
     *       readByte();
     *   }
     * }</pre>
     */
    public int readAnnotation() {
        checkType(ENCODED_ANNOTATION);
        type = MUST_READ;
        annotationType = Leb128.readUnsignedLeb128(in);
        return Leb128.readUnsignedLeb128(in);
    }

    /**
     * Returns the type of the annotation just returned by {@link
     * #readAnnotation()}. This method's value is undefined unless the most
     * recent call was to {@link #readAnnotation()}.
     */
    public int getAnnotationType() {
        return annotationType;
    }

    public int readAnnotationName() {
        return Leb128.readUnsignedLeb128(in);
    }

    public byte readByte() {
        checkType(ENCODED_BYTE);
        type = MUST_READ;
        return (byte) EncodedValueCodec.readSignedInt(in, arg);
    }

    public short readShort() {
        checkType(ENCODED_SHORT);
        type = MUST_READ;
        return (short) EncodedValueCodec.readSignedInt(in, arg);
    }

    public char readChar() {
        checkType(ENCODED_CHAR);
        type = MUST_READ;
        return (char) EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public int readInt() {
        checkType(ENCODED_INT);
        type = MUST_READ;
        return EncodedValueCodec.readSignedInt(in, arg);
    }

    public long readLong() {
        checkType(ENCODED_LONG);
        type = MUST_READ;
        return EncodedValueCodec.readSignedLong(in, arg);
    }

    public float readFloat() {
        checkType(ENCODED_FLOAT);
        type = MUST_READ;
        return Float.intBitsToFloat(EncodedValueCodec.readUnsignedInt(in, arg, true));
    }

    public double readDouble() {
        checkType(ENCODED_DOUBLE);
        type = MUST_READ;
        return Double.longBitsToDouble(EncodedValueCodec.readUnsignedLong(in, arg, true));
    }

    public int readMethodType() {
        checkType(ENCODED_METHOD_TYPE);
        type = MUST_READ;
        return EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public int readMethodHandle() {
        checkType(ENCODED_METHOD_HANDLE);
        type = MUST_READ;
        return EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public int readString() {
        checkType(ENCODED_STRING);
        type = MUST_READ;
        return EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public int readType() {
        checkType(ENCODED_TYPE);
        type = MUST_READ;
        return EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public int readField() {
        checkType(ENCODED_FIELD);
        type = MUST_READ;
        return EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public int readEnum() {
        checkType(ENCODED_ENUM);
        type = MUST_READ;
        return EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public int readMethod() {
        checkType(ENCODED_METHOD);
        type = MUST_READ;
        return EncodedValueCodec.readUnsignedInt(in, arg, false);
    }

    public void readNull() {
        checkType(ENCODED_NULL);
        type = MUST_READ;
    }

    public boolean readBoolean() {
        checkType(ENCODED_BOOLEAN);
        type = MUST_READ;
        return arg != 0;
    }

    /**
     * Skips a single value, including its nested values if it is an array or
     * annotation.
     */
    public void skipValue() {
        switch (peek()) {
        case ENCODED_BYTE:
            readByte();
            break;
        case ENCODED_SHORT:
            readShort();
            break;
        case ENCODED_CHAR:
            readChar();
            break;
        case ENCODED_INT:
            readInt();
            break;
        case ENCODED_LONG:
            readLong();
            break;
        case ENCODED_FLOAT:
            readFloat();
            break;
        case ENCODED_DOUBLE:
            readDouble();
            break;
        case ENCODED_METHOD_TYPE:
            readMethodType();
            break;
        case ENCODED_METHOD_HANDLE:
            readMethodHandle();
            break;
        case ENCODED_STRING:
            readString();
            break;
        case ENCODED_TYPE:
            readType();
            break;
        case ENCODED_FIELD:
            readField();
            break;
        case ENCODED_ENUM:
            readEnum();
            break;
        case ENCODED_METHOD:
            readMethod();
            break;
        case ENCODED_ARRAY:
            for (int i = 0, size = readArray(); i < size; i++) {
                skipValue();
            }
            break;
        case ENCODED_ANNOTATION:
            for (int i = 0, size = readAnnotation(); i < size; i++) {
                readAnnotationName();
                skipValue();
            }
            break;
        case ENCODED_NULL:
            readNull();
            break;
        case ENCODED_BOOLEAN:
            readBoolean();
            break;
        default:
            throw new DexException("Unexpected type: " + Integer.toHexString(type));
        }
    }

    private void checkType(int expected) {
        if (peek() != expected) {
            throw new IllegalStateException(
                    String.format("Expected %x but was %x", expected, peek()));
        }
    }
}
