// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import com.android.dex.Dex;
import com.android.dex.FieldId;
import com.android.dex.MethodId;
import com.android.dex.ProtoId;
import com.android.dex.TypeList;
import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableList;
import java.util.LinkedHashSet;

/**
 * Helper to track how many unique field and method references we've seen in a given set of .dex
 * files.
 */
class DexLimitTracker {

  private final LinkedHashSet<FieldDescriptor> fieldsSeen = new LinkedHashSet<>();
  private final LinkedHashSet<MethodDescriptor> methodsSeen = new LinkedHashSet<>();
  private final int maxNumberOfIdxPerDex;

  public DexLimitTracker(int maxNumberOfIdxPerDex) {
    this.maxNumberOfIdxPerDex = maxNumberOfIdxPerDex;
  }

  /**
   * Returns whether we're within limits.
   *
   * @return {@code true} if method or field references are outside limits, {@code false} both are
   *     within limits.
   */
  public boolean outsideLimits() {
    return fieldsSeen.size() > maxNumberOfIdxPerDex
        || methodsSeen.size() > maxNumberOfIdxPerDex;
  }

  public void clear() {
    fieldsSeen.clear();
    methodsSeen.clear();
  }

  public void track(Dex dexFile) {
    int fieldCount = dexFile.fieldIds().size();
    for (int fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex) {
      fieldsSeen.add(FieldDescriptor.fromDex(dexFile, fieldIndex));
    }
    int methodCount = dexFile.methodIds().size();
    for (int methodIndex = 0; methodIndex < methodCount; ++methodIndex) {
      methodsSeen.add(MethodDescriptor.fromDex(dexFile, methodIndex));
    }
  }

  private static String typeName(Dex dex, int typeIndex) {
    return dex.typeNames().get(typeIndex);
  }

  @AutoValue
  abstract static class FieldDescriptor {
    static FieldDescriptor fromDex(Dex dex, int fieldIndex) {
      FieldId field = dex.fieldIds().get(fieldIndex);
      String name = dex.strings().get(field.getNameIndex());
      String declaringClass = typeName(dex, field.getDeclaringClassIndex());
      String type = typeName(dex, field.getTypeIndex());
      return new AutoValue_DexLimitTracker_FieldDescriptor(declaringClass, name, type);
    }

    abstract String declaringClass();
    abstract String fieldName();
    abstract String fieldType();

    @Override
    @Memoized
    public abstract int hashCode();
  }

  @AutoValue
  abstract static class MethodDescriptor {
    static MethodDescriptor fromDex(Dex dex, int methodIndex) {
      MethodId method = dex.methodIds().get(methodIndex);
      ProtoId proto = dex.protoIds().get(method.getProtoIndex());
      String name = dex.strings().get(method.getNameIndex());
      String declaringClass = typeName(dex, method.getDeclaringClassIndex());
      String returnType = typeName(dex, proto.getReturnTypeIndex());
      TypeList parameterTypeIndices = dex.readTypeList(proto.getParametersOffset());
      ImmutableList.Builder<String> parameterTypes = ImmutableList.builder();
      for (short parameterTypeIndex : parameterTypeIndices.getTypes()) {
        parameterTypes.add(typeName(dex, parameterTypeIndex & 0xFFFF));
      }
      return new AutoValue_DexLimitTracker_MethodDescriptor(
          declaringClass, name, parameterTypes.build(), returnType);
    }

    abstract String declaringClass();
    abstract String methodName();
    abstract ImmutableList<String> parameterTypes();
    abstract String returnType();

    @Override
    @Memoized
    public abstract int hashCode();
  }
}
