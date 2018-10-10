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
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import javax.annotation.Nullable;

/**
 * Encapsulates the four syntactic variants of Skylark imports: Absolute paths, relative paths,
 * absolute labels, and relative labels.
 */
public interface SkylarkImport extends Serializable {
  /**
   * Returns the string originally used to specify the import (may represent a label or a path).
   */
  String getImportString();

  /**
   * Returns the import in the form of a path fragment for use by tools. Label-based imports are
   * converted to paths as follows: Imports using absolute labels or paths yield an absolute path
   * (whose root corresponds to the containing depot). Imports using relative labels yield a
   * package-relate path, and imports using relative paths yield a directory (of the importing-file)
   * relative path. All paths reference file names ending in '.bzl'. If there is an external
   * repository prefix, it is ignored.
   */
  PathFragment asPathFragment();

  /**
   * Given a {@link Label} representing the file that contains this import, returns a {@link Label}
   * representing the .bzl file to be imported.
   *
   * @throws IllegalStateException if this import takes the form of an absolute path.
   */
  Label getLabel(@Nullable Label containingFileLabel);
}
