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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

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
public interface JavaToolchainStarlarkApiProviderApi extends StructApi {

  String LEGACY_NAME = "java_toolchain";

  @StarlarkMethod(name = "source_version", doc = "The java source version.", structField = true)
  String getSourceVersion();

  @StarlarkMethod(name = "target_version", doc = "The java target version.", structField = true)
  String getTargetVersion();

  @StarlarkMethod(name = "label", doc = "The toolchain label.", structField = true)
  Label getToolchainLabel();

  @StarlarkMethod(
      name = "forcibly_disable_header_compilation",
      documented = false,
      useStarlarkThread = true)
  boolean getForciblyDisableHeaderCompilationStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "has_header_compiler", documented = false, useStarlarkThread = true)
  boolean hasHeaderCompiler(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "has_header_compiler_direct", documented = false, useStarlarkThread = true)
  boolean hasHeaderCompilerDirect(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "single_jar", doc = "The SingleJar deploy jar.", structField = true)
  FilesToRunProviderApi<? extends FileApi> getSingleJar();

  @Nullable
  @StarlarkMethod(
      name = "one_version_tool",
      doc = "The tool that enforces One-Version compliance of java binaries.",
      structField = true,
      allowReturnNones = true)
  FilesToRunProviderApi<? extends FileApi> getOneVersionBinary();

  @StarlarkMethod(
      name = "one_version_allowlist",
      doc = "The allowlist used by the One-Version compliance checker",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileApi getOneVersionAllowlist();

  @StarlarkMethod(
      name = "one_version_allowlist_for_tests",
      doc = "The allowlist used by the One-Version compliance checker to disable enforcement",
      useStarlarkThread = true,
      allowReturnNones = true)
  @Nullable
  FileApi getOneVersionAllowlistForTests(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "bootclasspath",
      doc = "The Java target bootclasspath entries. Corresponds to javac's -bootclasspath flag.",
      structField = true)
  Depset getStarlarkBootclasspath();

  @StarlarkMethod(
      name = "jvm_opt",
      doc = "The default options for the JVM running the java compiler and associated tools.",
      structField = true)
  Depset getStarlarkJvmOptions();

  @StarlarkMethod(
      name = "ijar",
      doc = "A FilesToRunProvider representing the ijar executable.",
      structField = true)
  FilesToRunProviderApi<?> getIjar();

  @StarlarkMethod(
      name = "jacocorunner",
      doc = "The jacocorunner used by the toolchain.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FilesToRunProviderApi<?> getJacocoRunner();

  @StarlarkMethod(name = "tools", doc = "The compilation tools.", structField = true)
  Depset getStarlarkTools();

  @StarlarkMethod(name = "java_runtime", doc = "The java runtime information.", structField = true)
  JavaRuntimeInfoApi getJavaRuntime();

  @StarlarkMethod(
      name = "android_linter",
      documented = false,
      useStarlarkThread = true,
      allowReturnNones = true)
  @Nullable
  StarlarkValue stalarkAndroidLinter(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "timezone_data",
      doc = "The latest timezone data resource jar that can be loaded by java binaries",
      useStarlarkThread = true,
      allowReturnNones = true)
  @Nullable
  FileApi getTimezoneDataForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "compatible_javacopts",
      doc = "Return the map of target environment-specific javacopts",
      parameters = {
        @Param(
            name = "key",
            allowedTypes = {
              @ParamType(type = String.class),
            },
            defaultValue = "")
      },
      allowReturnNones = true,
      useStarlarkThread = true)
  @Nullable
  ImmutableList<String> getCompatibleJavacOptionsForStarlark(String key, StarlarkThread thread)
      throws EvalException;

  @Nullable
  @StarlarkMethod(
      name = "deps_checker",
      documented = false,
      useStarlarkThread = true,
      allowReturnNones = true)
  FileApi getDepsCheckerForStarlark(StarlarkThread thread) throws EvalException;

  @Nullable
  @StarlarkMethod(
      name = "proguard_allowlister",
      doc = "Return the binary to validate proguard configuration",
      structField = true,
      allowReturnNones = true)
  FilesToRunProviderApi<? extends FileApi> getProguardAllowlister();

  @StarlarkMethod(name = "package_configuration", documented = false, useStarlarkThread = true)
  Sequence<?> getPackageConfigurationStarlark(StarlarkThread thread) throws EvalException;
}
