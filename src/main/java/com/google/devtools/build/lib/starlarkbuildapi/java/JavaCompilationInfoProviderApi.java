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

package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Info object for compilation information for java rules. */
@StarlarkBuiltin(
    name = "java_compilation_info",
    category = DocCategory.PROVIDER,
    doc = "Provides access to compilation information for Java rules.")
public interface JavaCompilationInfoProviderApi<FileT extends FileApi> extends StarlarkValue {

  @StarlarkMethod(name = "javac_options", structField = true, doc = "Options to java compiler.")
  ImmutableList<String> getJavacOpts();

  @StarlarkMethod(
      name = "runtime_classpath",
      structField = true,
      doc = "Run-time classpath for this Java target.")
  Depset /*<FileT>*/ getRuntimeClasspath();

  @StarlarkMethod(
      name = "compilation_classpath",
      structField = true,
      doc = "Compilation classpath for this Java target.")
  Depset /*<FileT>*/ getCompilationClasspath();

  @StarlarkMethod(
      name = "boot_classpath",
      structField = true,
      doc = "Boot classpath for this Java target.")
  ImmutableList<FileT> getBootClasspathList();
}
