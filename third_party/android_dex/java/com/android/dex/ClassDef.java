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

/**
 * A type definition.
 */
public final class ClassDef {
    public static final int NO_INDEX = -1;
    private final Dex buffer;
    private final int offset;
    private final int typeIndex;
    private final int accessFlags;
    private final int supertypeIndex;
    private final int interfacesOffset;
    private final int sourceFileIndex;
    private final int annotationsOffset;
    private final int classDataOffset;
    private final int staticValuesOffset;

    public ClassDef(Dex buffer, int offset, int typeIndex, int accessFlags,
            int supertypeIndex, int interfacesOffset, int sourceFileIndex,
            int annotationsOffset, int classDataOffset, int staticValuesOffset) {
        this.buffer = buffer;
        this.offset = offset;
        this.typeIndex = typeIndex;
        this.accessFlags = accessFlags;
        this.supertypeIndex = supertypeIndex;
        this.interfacesOffset = interfacesOffset;
        this.sourceFileIndex = sourceFileIndex;
        this.annotationsOffset = annotationsOffset;
        this.classDataOffset = classDataOffset;
        this.staticValuesOffset = staticValuesOffset;
    }

    public int getOffset() {
        return offset;
    }

    public int getTypeIndex() {
        return typeIndex;
    }

    public int getSupertypeIndex() {
        return supertypeIndex;
    }

    public int getInterfacesOffset() {
        return interfacesOffset;
    }

    public short[] getInterfaces() {
        return buffer.readTypeList(interfacesOffset).getTypes();
    }

    public int getAccessFlags() {
        return accessFlags;
    }

    public int getSourceFileIndex() {
        return sourceFileIndex;
    }

    public int getAnnotationsOffset() {
        return annotationsOffset;
    }

    public int getClassDataOffset() {
        return classDataOffset;
    }

    public int getStaticValuesOffset() {
        return staticValuesOffset;
    }

    @Override
    public String toString() {
        if (buffer == null) {
            return typeIndex + " " + supertypeIndex;
        }

        StringBuilder result = new StringBuilder();
        result.append(buffer.typeNames().get(typeIndex));
        if (supertypeIndex != NO_INDEX) {
            result.append(" extends ").append(buffer.typeNames().get(supertypeIndex));
        }
        return result.toString();
    }
}
