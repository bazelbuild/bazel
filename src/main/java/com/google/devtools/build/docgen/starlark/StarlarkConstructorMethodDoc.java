// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen.starlark;

import java.lang.reflect.Method;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Starlark;

/**
 * A class representing a Java method callable from Starlark which constructs a type of Starlark
 * object. Such a method is annotated with {@link StarlarkConstructor}, and has special handling.
 */
public final class StarlarkConstructorMethodDoc extends StarlarkMethodDoc {
  private final String fullyQualifiedName;

  public StarlarkConstructorMethodDoc(
      String fullyQualifiedName,
      Method javaMethod,
      StarlarkMethod annotation,
      StarlarkDocExpander expander) {
    super(javaMethod, annotation, expander);
    this.fullyQualifiedName = fullyQualifiedName;
  }

  @Override
  public String getName() {
    return fullyQualifiedName;
  }

  @Override
  public String getRawDocumentation() {
    return annotation.doc();
  }

  @Override
  public String getSignature() {
    return getSignature(fullyQualifiedName);
  }

  @Override
  public String getReturnType() {
    return Starlark.classType(javaMethod.getReturnType());
  }

  @Override
  public String toString() {
    return String.format(
        "StarlarkConstructorMethodDoc{fullyQualifiedName=%s method=%s callable=%s}",
        fullyQualifiedName, javaMethod, formatCallable());
  }

  private String formatCallable() {
    return String.format(
        "StarlarkMethod{name=%s selfCall=%s structField=%s doc=%s}",
        annotation.name(), annotation.selfCall(), annotation.structField(), annotation.doc());
  }
}
