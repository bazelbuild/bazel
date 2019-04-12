// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.cmdline.Label;
import java.io.Serializable;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Encapsulates the two syntactic variants of Starlark imports: absolute labels and relative labels.
 */
public abstract class SkylarkImport implements Serializable {
  private final String importString;

  protected SkylarkImport(String importString) {
    this.importString = importString;
  }

  /** Returns the string originally used to specify the import (represents a label). */
  public String getImportString() {
    return importString;
  }

  /**
   * Given a {@link Label} representing the file that contains this import, returns a {@link Label}
   * representing the .bzl file to be imported.
   *
   * @throws IllegalStateException if this import takes the form of an absolute path.
   */
  public abstract Label getLabel(@Nullable Label containingFileLabel);

  @Override
  public int hashCode() {
    return Objects.hash(getClass(), importString);
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }

    if (!(that instanceof SkylarkImport)) {
      return false;
    }

    return (that instanceof SkylarkImport)
        && Objects.equals(importString, ((SkylarkImport) that).importString);
  }
}
