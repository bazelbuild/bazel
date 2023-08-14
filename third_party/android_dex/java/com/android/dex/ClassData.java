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

public final class ClassData {
    private final Field[] staticFields;
    private final Field[] instanceFields;
    private final Method[] directMethods;
    private final Method[] virtualMethods;

    public ClassData(Field[] staticFields, Field[] instanceFields,
            Method[] directMethods, Method[] virtualMethods) {
        this.staticFields = staticFields;
        this.instanceFields = instanceFields;
        this.directMethods = directMethods;
        this.virtualMethods = virtualMethods;
    }

    public Field[] getStaticFields() {
        return staticFields;
    }

    public Field[] getInstanceFields() {
        return instanceFields;
    }

    public Method[] getDirectMethods() {
        return directMethods;
    }

    public Method[] getVirtualMethods() {
        return virtualMethods;
    }

    public Field[] allFields() {
        Field[] result = new Field[staticFields.length + instanceFields.length];
        System.arraycopy(staticFields, 0, result, 0, staticFields.length);
        System.arraycopy(instanceFields, 0, result, staticFields.length, instanceFields.length);
        return result;
    }

    public Method[] allMethods() {
        Method[] result = new Method[directMethods.length + virtualMethods.length];
        System.arraycopy(directMethods, 0, result, 0, directMethods.length);
        System.arraycopy(virtualMethods, 0, result, directMethods.length, virtualMethods.length);
        return result;
    }

    public static class Field {
        private final int fieldIndex;
        private final int accessFlags;

        public Field(int fieldIndex, int accessFlags) {
            this.fieldIndex = fieldIndex;
            this.accessFlags = accessFlags;
        }

        public int getFieldIndex() {
            return fieldIndex;
        }

        public int getAccessFlags() {
            return accessFlags;
        }
    }

    public static class Method {
        private final int methodIndex;
        private final int accessFlags;
        private final int codeOffset;

        public Method(int methodIndex, int accessFlags, int codeOffset) {
            this.methodIndex = methodIndex;
            this.accessFlags = accessFlags;
            this.codeOffset = codeOffset;
        }

        public int getMethodIndex() {
            return methodIndex;
        }

        public int getAccessFlags() {
            return accessFlags;
        }

        public int getCodeOffset() {
            return codeOffset;
        }
    }
}
