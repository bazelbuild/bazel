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
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.DependencyInfo;
import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.Locale;
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

  /** Represents a value that can be encoded into an int[] field initializer. */
  public interface IntArrayValue {
    public void pushValueOntoStack(InstructionAdapter insts);

    public String sourceRepresentation();
  }

  /** Represents an integer primitive. */
  public static class IntegerValue implements IntArrayValue {
    private final int value;

    public IntegerValue(int value) {
      this.value = value;
    }

    @Override
    public void pushValueOntoStack(InstructionAdapter insts) {
      insts.iconst(value);
    }

    @Override
    public String sourceRepresentation() {
      return String.format("0x%x", value);
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof IntegerValue) {
        IntegerValue other = (IntegerValue) obj;
        return value == other.value;
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Integer.hashCode(value);
    }

    @Override
    public String toString() {
      return Integer.toString(value);
    }
  }

  /** Represents an reference to a static field that holds an integer primitive. */
  public static class StaticIntFieldReference implements IntArrayValue {
    private final String className;
    private final String fieldName;

    private StaticIntFieldReference(String className, String fieldName) {
      this.className = className;
      this.fieldName = fieldName;
    }

    public static StaticIntFieldReference parse(String s) {
      final int fieldSep = s.lastIndexOf('.');
      if (fieldSep < 0) {
        throw new IllegalArgumentException("Unable to parse field reference from '" + s + "'");
      }
      final String className = s.substring(0, fieldSep);
      final String fieldName = s.substring(fieldSep + 1);
      if (className.isEmpty() || fieldName.isEmpty()) {
        throw new IllegalArgumentException(
            "Unable to extract class and field name from '" + s + "'");
      }
      return new StaticIntFieldReference(className, fieldName);
    }

    @Override
    public void pushValueOntoStack(InstructionAdapter insts) {
      // The syntax of class names that appear in class file structures differs from the syntax of
      // class names in source code.
      // See: https://docs.oracle.com/javase/specs/jvms/se7/html/jvms-4.html#jvms-4.2.1

      final List<String> parts = Splitter.on('.').splitToList(className);

      // If the class name ends in R.[type], replace the ending with R$[type] since [type] is a
      // class nested within the R class.
      final boolean replacePeriod = !Iterables.getLast(parts).startsWith("R$");

      final StringBuilder asmClassName = new StringBuilder();
      for (int i = 0, n = parts.size(); i < n; i++) {
        final String part = parts.get(i);
        asmClassName.append(part);
        if (i == n - 2 && replacePeriod && part.equals("R")) {
          asmClassName.append('$');
          continue;
        }
        if (i != n - 1) {
          // Replace all package seperating periods with forward slashes.
          asmClassName.append('/');
        }
      }

      insts.getstatic(asmClassName.toString(), fieldName, "I");
    }

    @Override
    public String sourceRepresentation() {
      return String.format(Locale.US, "%s.%s", className, fieldName);
    }

    @Override
    public int hashCode() {
      return Objects.hash(className, fieldName);
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof StaticIntFieldReference) {
        StaticIntFieldReference other = (StaticIntFieldReference) obj;
        return Objects.equals(className, other.className)
            && Objects.equals(fieldName, other.fieldName);
      }
      return false;
    }

    @Override
    public String toString() {
      return sourceRepresentation();
    }
  }

  private final DependencyInfo dependencyInfo;
  private final Visibility visibility;
  private final String fieldName;
  private final ImmutableList<IntArrayValue> values;

  private IntArrayFieldInitializer(
      DependencyInfo dependencyInfo,
      Visibility visibility,
      String fieldName,
      ImmutableList<IntArrayValue> values) {
    this.dependencyInfo = dependencyInfo;
    this.visibility = visibility;
    this.fieldName = fieldName;
    this.values = values;
  }

  public static FieldInitializer of(
      DependencyInfo dependencyInfo, Visibility visibility, String fieldName, String value) {
    Preconditions.checkArgument(value.startsWith("{ "), "Expected list starting with { ");
    Preconditions.checkArgument(value.endsWith(" }"), "Expected list ending with } ");
    String trimmedValue = value.substring(1, value.length() - 1).trim();
    // Check for an empty list.
    if (trimmedValue.isEmpty()) {
      return of(dependencyInfo, visibility, fieldName, ImmutableList.of());
    }
    ImmutableList.Builder<IntArrayValue> intValues = ImmutableList.builder();
    Iterable<String> valueStrings = Splitter.on(',').trimResults().split(trimmedValue);
    for (String valueString : valueStrings) {
      IntArrayValue elementValue;
      try {
        elementValue = StaticIntFieldReference.parse(valueString);
      } catch (IllegalArgumentException e) {
        elementValue = new IntegerValue(Integer.decode(valueString));
      }
      intValues.add(elementValue);
    }
    return of(dependencyInfo, visibility, fieldName, intValues.build());
  }

  public static IntArrayFieldInitializer of(
      DependencyInfo dependencyInfo,
      Visibility visibility,
      String fieldName,
      ImmutableList<IntArrayValue> values) {
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
  public void writeCLInit(InstructionAdapter insts, String className) {
    insts.iconst(values.size());
    insts.newarray(Type.INT_TYPE);
    int curIndex = 0;
    for (IntArrayValue value : values) {
      insts.dup();
      insts.iconst(curIndex);
      value.pushValueOntoStack(insts);
      insts.astore(Type.INT_TYPE);
      ++curIndex;
    }
    insts.putstatic(className, fieldName, DESC);
  }

  @Override
  public void writeInitSource(Writer writer, boolean finalFields) throws IOException {
    StringBuilder builder = new StringBuilder();
    boolean first = true;
    for (IntArrayValue value : values) {
      if (first) {
        first = false;
        builder.append(value.sourceRepresentation());
      } else {
        builder.append(String.format(Locale.US, ", %s", value.sourceRepresentation()));
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
  public int getMaxBytecodeSize() {
    // LDC_W(3)
    // NEWARRAY(2)
    //
    // 1..n
    // DUP(1)
    // LDC_W(3)
    // LDC_W|GETSTATIC(3)
    // IASTORE(1)
    //
    // PUTSTATIC(3)
    return 5 + values.size() * 8 + 3;
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
