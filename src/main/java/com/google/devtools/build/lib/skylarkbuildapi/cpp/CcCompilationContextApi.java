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
    documented = false,
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Immutable store of information needed for C++ compilation that is aggregated across "
            + "dependencies.")
public interface CcCompilationContextApi {
  @SkylarkCallable(
      name = "defines",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkDefines();

  @SkylarkCallable(
      name = "headers",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkHeaders();

  @SkylarkCallable(
      name = "system_includes",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkSystemIncludeDirs();

  @SkylarkCallable(
      name = "includes",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkIncludeDirs();

  @SkylarkCallable(
      name = "quote_includes",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkQuoteIncludeDirs();
}
