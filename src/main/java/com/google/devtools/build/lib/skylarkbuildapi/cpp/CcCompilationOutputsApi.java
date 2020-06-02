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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Interface for a structured representation of the compilation outputs of a C++ rule. */
@StarlarkBuiltin(
    name = "CcCompilationOutputs",
    category = StarlarkDocumentationCategory.BUILTIN,
    documented = true,
    doc = "Helper class containing CC compilation outputs.")
public interface CcCompilationOutputsApi<FileT extends FileApi> extends StarlarkValue {
  @StarlarkMethod(
      name = "objects",
      doc = "Non-PIC object files.",
      documented = true,
      structField = true)
  Sequence<FileT> getStarlarkObjects() throws EvalException;

  @StarlarkMethod(
      name = "pic_objects",
      doc = "PIC object files.",
      documented = true,
      structField = true)
  Sequence<FileT> getStarlarkPicObjects() throws EvalException;
}
