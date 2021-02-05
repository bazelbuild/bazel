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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ToolchainInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;

/**
 * Provides access to information about the Java toolchain rule. Accessible as a 'java_toolchain'
 * field on a Target struct.
 */
@StarlarkBuiltin(
    name = "JavaToolchainInfo",
    category = DocCategory.PROVIDER,
    doc =
        "Provides access to information about the Java toolchain rule. "
            + "Accessible as a 'java_toolchain' field on a Target struct.")
public interface JavaToolchainStarlarkApiProviderApi extends ToolchainInfoApi {

  String LEGACY_NAME = "java_toolchain";

  @StarlarkMethod(name = "source_version", doc = "The java source version.", structField = true)
  String getSourceVersion();

  @StarlarkMethod(name = "target_version", doc = "The java target version.", structField = true)
  String getTargetVersion();

  @StarlarkMethod(name = "single_jar", doc = "The SingleJar deploy jar.", structField = true)
  FileApi getSingleJar();

  @StarlarkMethod(
      name = "bootclasspath",
      doc = "The Java target bootclasspath entries. Corresponds to javac's -bootclasspath flag.",
      structField = true)
  Depset getStarlarkBootclasspath();

  @StarlarkMethod(
      name = "jvm_opt",
      doc = "The default options for the JVM running the java compiler and associated tools.",
      structField = true)
  Sequence<String> getStarlarkJvmOptions();

  @StarlarkMethod(
      name = "jacocorunner",
      doc = "The jacocorunner used by the toolchain.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FilesToRunProviderApi<?> getJacocoRunner();

  @StarlarkMethod(name = "tools", doc = "The compilation tools.", structField = true)
  Depset getStarlarkTools();
}
