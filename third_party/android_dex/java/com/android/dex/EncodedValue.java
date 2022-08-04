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

import com.android.dex.util.ByteArrayByteInput;
import com.android.dex.util.ByteInput;

/**
 * An encoded value or array.
 */
public final class EncodedValue implements Comparable<EncodedValue> {
    private final byte[] data;

    public EncodedValue(byte[] data) {
        this.data = data;
    }

    public ByteInput asByteInput() {
        return new ByteArrayByteInput(data);
    }

    public byte[] getBytes() {
        return data;
    }

    public void writeTo(Dex.Section out) {
        out.write(data);
    }

    @Override
    public int compareTo(EncodedValue other) {
        int size = Math.min(data.length, other.data.length);
        for (int i = 0; i < size; i++) {
            if (data[i] != other.data[i]) {
                return (data[i] & 0xff) - (other.data[i] & 0xff);
            }
        }
        return data.length - other.data.length;
    }

    @Override
    public String toString() {
        return Integer.toHexString(data[0] & 0xff) + "...(" + data.length + ")";
    }
}
