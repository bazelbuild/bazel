/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import org.objectweb.asm.Type;

/**
 * Represents the identifiable name of a Java class or interface with convenient conversions among
 * different names.
 */
@AutoValue
public abstract class ClassName implements TypeMappable<ClassName> {

  /**
   * The textual binary name used to index the class name, as defined at,
   * https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-4.html#jvms-4.2.1
   */
  public abstract String binaryName();

  public static ClassName create(String binaryName) {
    checkState(
        !binaryName.contains("."),
        "Expected a binary/internal class name ('/'-delimited) instead of a qualified name."
            + " Actual: (%s)",
        binaryName);
    return new AutoValue_ClassName(binaryName);
  }

  public static ClassName create(Class<?> clazz) {
    return create(Type.getType(clazz));
  }

  public static ClassName create(Type asmType) {
    return create(asmType.getInternalName());
  }

  public final Type toAsmObjectType() {
    return Type.getObjectType(binaryName());
  }

  public final String qualifiedName() {
    return binaryName().replace('/', '.');
  }

  public ClassName innerClass(String innerClassSimpleName) {
    return ClassName.create(binaryName() + '$' + innerClassSimpleName);
  }

  public final String simpleName() {
    String binaryName = binaryName();
    int i = binaryName.lastIndexOf('/');
    return i < 0 ? binaryName : binaryName.substring(i + 1);
  }

  public final String classFilePathName() {
    return binaryName() + ".class";
  }

  public final ClassName prependPrefix(String prefix) {
    return ClassName.create(prefix + binaryName());
  }

  @Override
  public ClassName acceptTypeMapper(TypeMapper typeMapper) {
    return typeMapper.map(this);
  }
}
