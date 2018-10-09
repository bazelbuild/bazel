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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Parameters that affect linking actions.
 *
 * <p>The parameters concerned are the link options (strings) passed to the linker, linkstamps, a
 * list of libraries to be linked in, and a list of libraries to build at link time.
 *
 * <p>Items in the collections are stored in nested sets. Link options and libraries are stored in
 * link order (preorder) and linkstamps are sorted.
 */
@SkylarkModule(
    name = "CcLinkParams",
    documented = false,
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Parameters that affect linking actions.")
public interface CcLinkParamsApi {
  @SkylarkCallable(
      name = "user_link_flags",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkLinkopts();

  @SkylarkCallable(
      name = "libraries_to_link",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkLibrariesToLink();

  @SkylarkCallable(
      name = "dynamic_libraries_for_runtime",
      documented = false,
      allowReturnNones = true,
      structField = true)
  SkylarkNestedSet getSkylarkDynamicLibrariesForRuntime();
}
