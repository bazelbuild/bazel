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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Interface for a structured representation of the compilation outputs of a C++ rule. */
@StarlarkBuiltin(
    name = "CcCompilationOutputs",
    category = DocCategory.BUILTIN,
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

  @StarlarkMethod(name = "temps", documented = false, useStarlarkThread = true)
  Depset getStarlarkTemps(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "_header_tokens", documented = false, structField = true)
  Sequence<FileT> getStarlarkHeaderTokens() throws EvalException;

  @StarlarkMethod(name = "_module_files", documented = false, structField = true)
  Sequence<FileT> getStarlarkModuleFiles() throws EvalException;

  @StarlarkMethod(name = "_lto_compilation_context", documented = false, structField = true)
  Object getLtoCompilationContextForStarlark() throws EvalException;

  @StarlarkMethod(name = "_dwo_files", documented = false, structField = true)
  Sequence<FileT> getStarlarkDwoFiles() throws EvalException;

  @StarlarkMethod(name = "_pic_dwo_files", documented = false, structField = true)
  Sequence<FileT> getStarlarkPicDwoFiles() throws EvalException;

  @StarlarkMethod(name = "_gcno_files", documented = false, structField = true)
  Sequence<FileT> getStarlarkGcnoFiles() throws EvalException;

  @StarlarkMethod(name = "_pic_gcno_files", documented = false, structField = true)
  Sequence<FileT> getStarlarkPicGcnoFiles() throws EvalException;
}
