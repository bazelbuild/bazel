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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/**
 * Interface for a store of information needed for C++ compilation aggregated across dependencies.
 */
@SkylarkModule(
    name = "CompilationContext",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Immutable store of information needed for C++ compilation that is aggregated across "
            + "dependencies.")
public interface CcCompilationContextApi<FileT extends FileApi> extends StarlarkValue {
  @SkylarkCallable(
      name = "defines",
      doc =
          "Returns the set of defines needed to compile this target. Each define is a string."
              + " These values are propagated to the target's transitive dependencies.",
      structField = true)
  Depset getSkylarkDefines();

  @SkylarkCallable(
      name = "local_defines",
      doc =
          "Returns the set of defines needed to compile this target. Each define is a string."
              + " These values are not propagated to the target's transitive dependencies.",
      structField = true)
  Depset getSkylarkNonTransitiveDefines();

  @SkylarkCallable(
      name = "headers",
      doc = "Returns the set of headers needed to compile this target.",
      structField = true)
  Depset getSkylarkHeaders();

  @SkylarkCallable(
      name = "system_includes",
      doc =
          "Returns the set of search paths (as strings) for header files referenced by angle"
              + " brackets, e.g. #include &lt;foo/bar/header.h&gt;. They can be either relative to"
              + " the exec root or absolute. Usually passed with -isystem.",
      structField = true)
  Depset getSkylarkSystemIncludeDirs();

  @SkylarkCallable(
      name = "framework_includes",
      doc =
          "Returns the set of search paths (as strings) for framework header files. Usually passed"
              + " with -F.",
      structField = true)
  Depset getSkylarkFrameworkIncludeDirs();

  @SkylarkCallable(
      name = "includes",
      doc =
          "Returns the set of search paths (as strings) for header files referenced both by angle"
              + " bracket and quotes. Usually passed with -I.",
      structField = true)
  Depset getSkylarkIncludeDirs();

  @SkylarkCallable(
      name = "quote_includes",
      doc =
          "Returns the set of search paths (as strings) for header files referenced by quotes,"
              + " e.g. #include \"foo/bar/header.h\". They can be either relative to the exec root"
              + " or absolute. Usually passed with -iquote.",
      structField = true)
  Depset getSkylarkQuoteIncludeDirs();

  @SkylarkCallable(
      name = "direct_headers",
      doc =
          "Returns the list of header files that are declared by the \"hdrs\" attribute of this"
              + " target.",
      structField = true)
  StarlarkList<FileT> getSkylarkDirectModularHeaders();

  @SkylarkCallable(
      name = "direct_textual_headers",
      doc =
          "Returns the list of header files that are declared by the \"textual_hdrs\" attribute of"
              + " this target.",
      structField = true)
  StarlarkList<FileT> getSkylarkDirectTextualHeaders();
}
