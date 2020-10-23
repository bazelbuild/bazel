// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.resources;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.DependencyInfo;
import java.io.IOException;
import java.io.Writer;
import java.util.Objects;
import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.InstructionAdapter;

/** Models an int[] field initializer. */
public final class IntArrayFieldInitializer implements FieldInitializer {

  private static final String DESC = "[I";

  private final DependencyInfo dependencyInfo;
  private final Visibility visibility;
  private final String fieldName;
  private final ImmutableList<Integer> values;

  private IntArrayFieldInitializer(
      DependencyInfo dependencyInfo,
      Visibility visibility,
      String fieldName,
      ImmutableList<Integer> values) {
    this.dependencyInfo = dependencyInfo;
    this.visibility = visibility;
    this.fieldName = fieldName;
    this.values = values;
  }

  public static FieldInitializer of(
      DependencyInfo dependencyInfo, Visibility visibility, String fieldName, String value) {
    Preconditions.checkArgument(value.startsWith("{ "), "Expected list starting with { ");
    Preconditions.checkArgument(value.endsWith(" }"), "Expected list ending with } ");
    // Check for an empty list, which is "{ }".
    if (value.length() < 4) {
      return of(dependencyInfo, visibility, fieldName, ImmutableList.of());
    }
    ImmutableList.Builder<Integer> intValues = ImmutableList.builder();
    String trimmedValue = value.substring(2, value.length() - 2);
    Iterable<String> valueStrings = Splitter.on(',').trimResults().split(trimmedValue);
    for (String valueString : valueStrings) {
      intValues.add(Integer.decode(valueString));
    }
    return of(dependencyInfo, visibility, fieldName, intValues.build());
  }

  public static IntArrayFieldInitializer of(
      DependencyInfo dependencyInfo,
      Visibility visibility,
      String fieldName,
      ImmutableList<Integer> values) {
    return new IntArrayFieldInitializer(dependencyInfo, visibility, fieldName, values);
  }

  @Override
  public boolean writeFieldDefinition(
      ClassWriter cw, boolean isFinal, boolean annotateTransitiveFields) {
    int accessLevel = Opcodes.ACC_STATIC;
    if (visibility != Visibility.PRIVATE) {
      accessLevel |= Opcodes.ACC_PUBLIC;
    }
    if (isFinal) {
      accessLevel |= Opcodes.ACC_FINAL;
    }
    FieldVisitor fv = cw.visitField(accessLevel, fieldName, DESC, null, null);
    if (annotateTransitiveFields
        && dependencyInfo.dependencyType() == DependencyInfo.DependencyType.TRANSITIVE) {
      AnnotationVisitor av =
          fv.visitAnnotation(
              RClassGenerator.PROVENANCE_ANNOTATION_CLASS_DESCRIPTOR, /*visible=*/ true);
      av.visit(RClassGenerator.PROVENANCE_ANNOTATION_LABEL_KEY, dependencyInfo.label());
      av.visitEnd();
    }
    fv.visitEnd();
    return true;
  }

  @Override
  public int writeCLInit(InstructionAdapter insts, String className) {
    insts.iconst(values.size());
    insts.newarray(Type.INT_TYPE);
    int curIndex = 0;
    for (Integer value : values) {
      insts.dup();
      insts.iconst(curIndex);
      insts.iconst(value);
      insts.astore(Type.INT_TYPE);
      ++curIndex;
    }
    insts.putstatic(className, fieldName, DESC);
    // Needs up to 4 stack slots for: the array ref for the putstatic, the dup of the array ref
    // for the store, the index, and the value to store.
    return 4;
  }

  @Override
  public void writeInitSource(Writer writer, boolean finalFields) throws IOException {
    StringBuilder builder = new StringBuilder();
    boolean first = true;
    for (Integer attrId : values) {
      if (first) {
        first = false;
        builder.append(String.format("0x%x", attrId));
      } else {
        builder.append(String.format(", 0x%x", attrId));
      }
    }

    writer.write(
        String.format(
            "        %s static %sint[] %s = { %s };\n",
            visibility != Visibility.PRIVATE ? "public" : "",
            finalFields ? "final " : "",
            fieldName,
            builder));
  }

  @Override
  public String getFieldName() {
    return fieldName;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("values", values).toString();
  }

  @Override
  public int hashCode() {
    return values.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof IntArrayFieldInitializer) {
      IntArrayFieldInitializer other = (IntArrayFieldInitializer) obj;
      return Objects.equals(dependencyInfo, other.dependencyInfo)
          && Objects.equals(fieldName, other.fieldName)
          && Objects.equals(values, other.values);
    }
    return false;
  }
}
