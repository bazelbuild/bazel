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

package com.android.dex.util;

/**
 * Unsigned arithmetic over Java's signed types.
 */
public final class Unsigned {
    private Unsigned() {}

    public static int compare(short ushortA, short ushortB) {
        if (ushortA == ushortB) {
            return 0;
        }
        int a = ushortA & 0xFFFF;
        int b = ushortB & 0xFFFF;
        return a < b ? -1 : 1;
    }

    public static int compare(int uintA, int uintB) {
        if (uintA == uintB) {
            return 0;
        }
        long a = uintA & 0xFFFFFFFFL;
        long b = uintB & 0xFFFFFFFFL;
        return a < b ? -1 : 1;
    }
}
