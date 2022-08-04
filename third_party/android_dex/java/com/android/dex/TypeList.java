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

import com.android.dex.util.Unsigned;

public final class TypeList implements Comparable<TypeList> {

    public static final TypeList EMPTY = new TypeList(null, Dex.EMPTY_SHORT_ARRAY);

    private final Dex dex;
    private final short[] types;

    public TypeList(Dex dex, short[] types) {
        this.dex = dex;
        this.types = types;
    }

    public short[] getTypes() {
        return types;
    }

    @Override
    public int compareTo(TypeList other) {
        for (int i = 0; i < types.length && i < other.types.length; i++) {
            if (types[i] != other.types[i]) {
                return Unsigned.compare(types[i], other.types[i]);
            }
        }
        return Unsigned.compare(types.length, other.types.length);
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("(");
        for (int i = 0, typesLength = types.length; i < typesLength; i++) {
            result.append(dex != null ? dex.typeNames().get(types[i]) : types[i]);
        }
        result.append(")");
        return result.toString();
    }
}
