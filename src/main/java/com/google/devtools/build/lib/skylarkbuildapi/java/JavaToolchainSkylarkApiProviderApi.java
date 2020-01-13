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

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainInfoApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Sequence;

/**
 * Provides access to information about the Java toolchain rule. Accessible as a 'java_toolchain'
 * field on a Target struct.
 */
@SkylarkModule(
    name = "JavaToolchainInfo",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Provides access to information about the Java toolchain rule. "
            + "Accessible as a 'java_toolchain' field on a Target struct.")
public interface JavaToolchainSkylarkApiProviderApi extends ToolchainInfoApi {

  String LEGACY_NAME = "java_toolchain";

  @SkylarkCallable(name = "source_version", doc = "The java source version.", structField = true)
  String getSourceVersion();

  @SkylarkCallable(name = "target_version", doc = "The java target version.", structField = true)
  String getTargetVersion();

  @SkylarkCallable(
      name = "javac_jar",
      doc = "The javac jar.",
      structField = true,
      allowReturnNones = true)
  FileApi getJavacJar();

  @SkylarkCallable(name = "single_jar", doc = "The SingleJar deploy jar.", structField = true)
  FileApi getSingleJar();

  @SkylarkCallable(
      name = "bootclasspath",
      doc = "The Java target bootclasspath entries. Corresponds to javac's -bootclasspath flag.",
      structField = true)
  Depset getSkylarkBootclasspath();

  @SkylarkCallable(
      name = "jvm_opt",
      doc = "The default options for the JVM running the java compiler and associated tools.",
      structField = true)
  Sequence<String> getSkylarkJvmOptions();

  @SkylarkCallable(name = "tools", doc = "The compilation tools.", structField = true)
  Depset getSkylarkTools();
}
