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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Interface for a configuration object which holds information about the build environment. */
@StarlarkBuiltin(
    name = "configuration",
    category = DocCategory.BUILTIN,
    doc =
        "This object holds information about the environment in which the build is running. See the"
            + " <a href='https://bazel.build/extending/rules#configurations'>Rules page</a> for"
            + " more on the general concept of configurations.")
public interface BuildConfigurationApi extends StarlarkValue {

  @StarlarkMethod(name = "bin_dir", structField = true, documented = false)
  @Deprecated
  FileRootApi getBinDir();

  @StarlarkMethod(name = "genfiles_dir", structField = true, documented = false)
  @Deprecated
  FileRootApi getGenfilesDir();

  @StarlarkMethod(
      name = "host_path_separator",
      structField = true,
      doc = "Returns the separator for PATH environment variable, which is ':' on Unix.")
  String getHostPathSeparator();

  @StarlarkMethod(
      name = "default_shell_env",
      structField = true,
      doc =
          "A dictionary representing the static local shell environment. It maps variables "
              + "to their values (strings).")
  @Deprecated // Use getActionEnvironment instead.
  ImmutableMap<String, String> getLocalShellEnvironment();

  @StarlarkMethod(
      name = "test_env",
      structField = true,
      doc =
          "A dictionary containing user-specified test environment variables and their values, as"
              + " set by the --test_env options. DO NOT USE! This is not the complete"
              + " environment!")
  ImmutableMap<String, String> getTestEnv();

  @StarlarkMethod(
      name = "coverage_enabled",
      structField = true,
      doc =
          "A boolean that tells whether code coverage is enabled for this run. Note that this does"
              + " not compute whether a specific rule should be instrumented for code coverage data"
              + " collection. For that, see the <a"
              + " href=\"../builtins/ctx.html#coverage_instrumented\"><code>ctx.coverage_instrumented</code></a>"
              + " function.")
  boolean isCodeCoverageEnabled();

  @StarlarkMethod(name = "stamp_binaries", documented = false, useStarlarkThread = true)
  boolean stampBinariesForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "is_tool_configuration", documented = false, useStarlarkThread = true)
  boolean isToolConfigurationForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "has_separate_genfiles_directory",
      documented = false,
      useStarlarkThread = true)
  boolean hasSeparateGenfilesDirectoryForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "is_sibling_repository_layout",
      documented = false,
      useStarlarkThread = true)
  boolean isSiblingRepositoryLayoutForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "runfiles_enabled", documented = false, useStarlarkThread = true)
  boolean runfilesEnabledForStarlark(StarlarkThread thread) throws EvalException;
}
