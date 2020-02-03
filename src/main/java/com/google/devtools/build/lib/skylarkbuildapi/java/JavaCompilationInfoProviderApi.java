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

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Info object for compilation information for java rules. */
@SkylarkModule(
    name = "java_compilation_info",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "Provides access to compilation information for Java rules.")
public interface JavaCompilationInfoProviderApi<FileT extends FileApi> extends StarlarkValue {

  @SkylarkCallable(name = "javac_options", structField = true, doc = "Options to java compiler.")
  ImmutableList<String> getJavacOpts();

  @SkylarkCallable(
      name = "runtime_classpath",
      structField = true,
      doc = "Run-time classpath for this Java target.")
  Depset /*<FileT>*/ getRuntimeClasspath();

  @SkylarkCallable(
      name = "compilation_classpath",
      structField = true,
      doc = "Compilation classpath for this Java target.")
  Depset /*<FileT>*/ getCompilationClasspath();

  @SkylarkCallable(
      name = "boot_classpath",
      structField = true,
      doc = "Boot classpath for this Java target.")
  ImmutableList<FileT> getBootClasspath();
}
