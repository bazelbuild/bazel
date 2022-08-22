/*
 * Copyright (C) 2017 The Android Open Source Project
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

import com.android.dex.Dex.Section;
import com.android.dex.util.Unsigned;

/**
 * A method_handle_item:
 * https://source.android.com/devices/tech/dalvik/dex-format#method-handle-item
 */
public class MethodHandle implements Comparable<MethodHandle> {

    /**
     * A method handle type code:
     * https://source.android.com/devices/tech/dalvik/dex-format#method-handle-type-codes
     */
    public enum MethodHandleType {
        METHOD_HANDLE_TYPE_STATIC_PUT(0x00),
        METHOD_HANDLE_TYPE_STATIC_GET(0x01),
        METHOD_HANDLE_TYPE_INSTANCE_PUT(0x02),
        METHOD_HANDLE_TYPE_INSTANCE_GET(0x03),
        METHOD_HANDLE_TYPE_INVOKE_STATIC(0x04),
        METHOD_HANDLE_TYPE_INVOKE_INSTANCE(0x05),
        METHOD_HANDLE_TYPE_INVOKE_DIRECT(0x06),
        METHOD_HANDLE_TYPE_INVOKE_CONSTRUCTOR(0x07),
        METHOD_HANDLE_TYPE_INVOKE_INTERFACE(0x08);

        private final int value;

        MethodHandleType(int value) {
            this.value = value;
        }

        static MethodHandleType fromValue(int value) {
            for (MethodHandleType methodHandleType : values()) {
                if (methodHandleType.value == value) {
                    return methodHandleType;
                }
            }
            throw new IllegalArgumentException(String.valueOf(value));
        }

        public boolean isField() {
            switch (this) {
                case METHOD_HANDLE_TYPE_STATIC_PUT:
                case METHOD_HANDLE_TYPE_STATIC_GET:
                case METHOD_HANDLE_TYPE_INSTANCE_PUT:
                case METHOD_HANDLE_TYPE_INSTANCE_GET:
                    return true;
                default:
                    return false;
            }
        }
    }

    private final Dex dex;
    private final MethodHandleType methodHandleType;
    private final int unused1;
    private final int fieldOrMethodId;
    private final int unused2;

    public MethodHandle(
            Dex dex,
            MethodHandleType methodHandleType,
            int unused1,
            int fieldOrMethodId,
            int unused2) {
        this.dex = dex;
        this.methodHandleType = methodHandleType;
        this.unused1 = unused1;
        this.fieldOrMethodId = fieldOrMethodId;
        this.unused2 = unused2;
    }

    @Override
    public int compareTo(MethodHandle o) {
        if (methodHandleType != o.methodHandleType) {
            return methodHandleType.compareTo(o.methodHandleType);
        }
        return Unsigned.compare(fieldOrMethodId, o.fieldOrMethodId);
    }

    public MethodHandleType getMethodHandleType() {
        return methodHandleType;
    }

    public int getUnused1() {
        return unused1;
    }

    public int getFieldOrMethodId() {
        return fieldOrMethodId;
    }

    public int getUnused2() {
        return unused2;
    }

    public void writeTo(Section out) {
        out.writeUnsignedShort(methodHandleType.value);
        out.writeUnsignedShort(unused1);
        out.writeUnsignedShort(fieldOrMethodId);
        out.writeUnsignedShort(unused2);
    }

    @Override
    public String toString() {
        if (dex == null) {
            return methodHandleType + " " + fieldOrMethodId;
        }
        return methodHandleType
                + " "
                + (methodHandleType.isField()
                        ? dex.fieldIds().get(fieldOrMethodId)
                        : dex.methodIds().get(fieldOrMethodId));
    }
}
