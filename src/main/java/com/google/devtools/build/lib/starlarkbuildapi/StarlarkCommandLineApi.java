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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/** Interface for a module associated with creating efficient command lines. */
@StarlarkBuiltin(
    name = "cmd_helper",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc = "Deprecated. Module for creating memory efficient command lines.")
public interface StarlarkCommandLineApi extends StarlarkValue {

  @StarlarkMethod(
      name = "join_paths",
      doc =
          "Deprecated. Creates a single command line argument joining the paths of a set "
              + "of files on the separator string.",
      parameters = {
        @Param(name = "separator", doc = "the separator string to join on."),
        @Param(name = "files", doc = "the files to concatenate.")
      })
  String joinPaths(String separator, Depset files) throws EvalException;
}
