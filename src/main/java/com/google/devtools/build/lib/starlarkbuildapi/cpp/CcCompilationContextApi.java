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
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/**
 * Interface for a store of information needed for C++ compilation aggregated across dependencies.
 */
@StarlarkBuiltin(
    name = "CompilationContext",
    category = DocCategory.PROVIDER,
    doc =
        "Immutable store of information needed for C++ compilation that is aggregated across "
            + "dependencies.")
public interface CcCompilationContextApi<FileT extends FileApi> extends StarlarkValue {
  @StarlarkMethod(
      name = "defines",
      doc =
          "Returns the set of defines needed to compile this target. Each define is a string."
              + " These values are propagated to the target's transitive dependencies.",
      structField = true)
  Depset getStarlarkDefines();

  @StarlarkMethod(
      name = "local_defines",
      doc =
          "Returns the set of defines needed to compile this target. Each define is a string."
              + " These values are not propagated to the target's transitive dependencies.",
      structField = true)
  Depset getStarlarkNonTransitiveDefines();

  @StarlarkMethod(
      name = "headers",
      doc = "Returns the set of headers needed to compile this target.",
      structField = true)
  Depset getStarlarkHeaders();

  @StarlarkMethod(
      name = "system_includes",
      doc =
          "Returns the set of search paths (as strings) for header files referenced by angle"
              + " brackets, e.g. #include &lt;foo/bar/header.h&gt;. They can be either relative to"
              + " the exec root or absolute. Usually passed with -isystem.",
      structField = true)
  Depset getStarlarkSystemIncludeDirs();

  @StarlarkMethod(
      name = "framework_includes",
      doc =
          "Returns the set of search paths (as strings) for framework header files. Usually passed"
              + " with -F.",
      structField = true)
  Depset getStarlarkFrameworkIncludeDirs();

  @StarlarkMethod(
      name = "includes",
      doc =
          "Returns the set of search paths (as strings) for header files referenced both by angle"
              + " bracket and quotes. Usually passed with -I.",
      structField = true)
  Depset getStarlarkIncludeDirs();

  @StarlarkMethod(
      name = "quote_includes",
      doc =
          "Returns the set of search paths (as strings) for header files referenced by quotes,"
              + " e.g. #include \"foo/bar/header.h\". They can be either relative to the exec root"
              + " or absolute. Usually passed with -iquote.",
      structField = true)
  Depset getStarlarkQuoteIncludeDirs();

  @StarlarkMethod(
      name = "direct_headers",
      doc =
          "Returns the list of modular headers that are declared by this target. This includes both"
              + " public headers (such as those listed in \"hdrs\") and private headers (such as"
              + " those listed in \"srcs\").",
      structField = true)
  StarlarkList<FileT> getStarlarkDirectModularHeaders();

  @StarlarkMethod(
      name = "direct_public_headers",
      doc =
          "Returns the list of modular public headers (those listed in \"hdrs\") that are declared"
              + " by this target.",
      structField = true)
  StarlarkList<FileT> getStarlarkDirectPublicHeaders();

  @StarlarkMethod(
      name = "direct_private_headers",
      doc =
          "Returns the list of modular private headers (those listed in \"srcs\") that are"
              + " declared by this target.",
      structField = true)
  StarlarkList<FileT> getStarlarkDirectPrivateHeaders();

  @StarlarkMethod(
      name = "direct_textual_headers",
      doc = "Returns the list of textual headers that are declared by this target.",
      structField = true)
  StarlarkList<FileT> getStarlarkDirectTextualHeaders();

  @StarlarkMethod(
      name = "transitive_compilation_prerequisites",
      documented = false,
      useStarlarkThread = true)
  Depset getStarlarkTransitiveCompilationPrerequisites(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "validation_artifacts",
      doc = "Returns the set of validation artifacts.",
      structField = true)
  Depset getStarlarkValidationArtifacts();
}
