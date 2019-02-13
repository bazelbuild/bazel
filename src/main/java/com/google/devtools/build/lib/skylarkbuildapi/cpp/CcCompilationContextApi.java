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

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Interface for a store of information needed for C++ compilation aggregated across dependencies.
 */
@SkylarkModule(
    name = "CompilationContext",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Immutable store of information needed for C++ compilation that is aggregated across "
            + "dependencies.")
public interface CcCompilationContextApi {
  @SkylarkCallable(
      name = "defines",
      doc = "Returns the set of defines needed to compile this target. Each define is a string.",
      structField = true)
  SkylarkNestedSet getSkylarkDefines();

  @SkylarkCallable(
      name = "headers",
      doc = "Returns the set of headers needed to compile this target.",
      structField = true)
  SkylarkNestedSet getSkylarkHeaders();

  @SkylarkCallable(
      name = "system_includes",
      doc =
          "Returns the set of search paths for header files referenced by angle brackets, e.g. "
              + "#include <foo/bar/header.h>. They can be either relative to the exec root "
              + "or absolute. Usually passed with -isystem.",
      structField = true)
  SkylarkNestedSet getSkylarkSystemIncludeDirs();

  @SkylarkCallable(
      name = "includes",
      doc =
          "Returns the set of search paths for header files referenced both by angle bracket and "
              + "quotes Usually passed with -I.",
      structField = true)
  SkylarkNestedSet getSkylarkIncludeDirs();

  @SkylarkCallable(
      name = "quote_includes",
      doc =
          "Returns the set of search paths for header files referenced by quotes, e.g. "
              + "#include \"foo/bar/header.h\". They can be either relative to the exec "
              + "root or absolute. Usually passed with -iquote.",
      structField = true)
  SkylarkNestedSet getSkylarkQuoteIncludeDirs();
}
