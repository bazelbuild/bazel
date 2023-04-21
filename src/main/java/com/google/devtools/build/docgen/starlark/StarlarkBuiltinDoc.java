// Copyright 2023 The Bazel Authors. All rights reserved.
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

import net.starlark.java.annot.StarlarkBuiltin;

/** A documentation page for a Starlark builtin type. */
public final class StarlarkBuiltinDoc extends StarlarkDocPage {
  private static final String SOURCE_ROOT = "src/main/java";

  private final StarlarkBuiltin starlarkBuiltin;
  private final Class<?> classObject;

  public StarlarkBuiltinDoc(
      StarlarkBuiltin starlarkBuiltin, Class<?> classObject, StarlarkDocExpander expander) {
    super(expander);
    this.starlarkBuiltin = starlarkBuiltin;
    this.classObject = classObject;
  }

  @Override
  public String getName() {
    return starlarkBuiltin.name();
  }

  @Override
  public String getRawDocumentation() {
    return starlarkBuiltin.doc();
  }

  @Override
  public String getTitle() {
    return starlarkBuiltin.name();
  }

  public Class<?> getClassObject() {
    return classObject;
  }

  @Override
  public String getSourceFile() {
    String[] parts = classObject.getName().split("\\$", -1);
    return String.format("%s/%s.java", SOURCE_ROOT, parts[0].replace('.', '/'));
  }
}
