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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * Object with information about C++ rules. Every C++-related target should provide this.
 */
@SkylarkModule(
  name = "CcSkylarkApiProvider",
  category = SkylarkModuleCategory.PROVIDER,
  doc =
      "Provides access to information about C++ rules.  "
          + "Every C++-related target provides this struct, accessible as a <code>cc</code> field "
          + "on <a href=\"Target.html\">target</a>."
)
public interface CcSkylarkApiProviderApi<FileT extends FileApi> {

  @SkylarkCallable(
      name = "transitive_headers",
      structField = true,
      doc =
          "Returns a <a href=\"depset.html\">depset</a> of headers that have been declared in the "
              + " <code>src</code> or <code>headers</code> attribute"
              + "(possibly empty but never <code>None</code>).")
  public NestedSet<FileT> getTransitiveHeaders();

  @SkylarkCallable(
      name = "libs",
      structField = true,
      doc =
          "Returns the <a href=\"depset.html\">depset</a> of libraries for either "
              + "<code>FULLY STATIC</code> mode (<code>linkopts=[\"-static\"]</code>) or "
              + "<code>MOSTLY STATIC</code> mode (<code>linkstatic=1</code>) "
              + "(possibly empty but never <code>None</code>)")
  public NestedSet<FileT> getLibraries();

  @SkylarkCallable(
      name = "link_flags",
      structField = true,
      doc =
          "Returns the list of flags given to the C++ linker command for either "
              + "<code>FULLY STATIC</code> mode (<code>linkopts=[\"-static\"]</code>) or "
              + "<code>MOSTLY STATIC</code> mode (<code>linkstatic=1</code>) "
              + "(possibly empty but never <code>None</code>)")
  public ImmutableList<String> getLinkopts();

  @SkylarkCallable(
      name = "defines",
      structField = true,
      doc =
          "Returns the list of defines used to compile this target "
              + "(possibly empty but never <code>None</code>).")
  public ImmutableList<String> getDefines();

  @SkylarkCallable(
      name = "system_include_directories",
      structField = true,
      doc =
          "Returns the list of system include directories used to compile this target "
              + "(possibly empty but never <code>None</code>).")
  public ImmutableList<String> getSystemIncludeDirs();

  @SkylarkCallable(
      name = "include_directories",
      structField = true,
      doc =
          "Returns the list of include directories used to compile this target "
              + "(possibly empty but never <code>None</code>).")
  public ImmutableList<String> getIncludeDirs();

  @SkylarkCallable(
      name = "quote_include_directories",
      structField = true,
      doc =
          "Returns the list of quote include directories used to compile this target "
              + "(possibly empty but never <code>None</code>).")
  public ImmutableList<String> getQuoteIncludeDirs();

  @SkylarkCallable(
      name = "compile_flags",
      structField = true,
      doc =
          "Returns the list of flags used to compile this target "
              + "(possibly empty but never <code>None</code>).")
  public ImmutableList<String> getCcFlags();
}
