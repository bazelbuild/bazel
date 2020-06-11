// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainInfoApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Information about the Java runtime being used. */
@StarlarkBuiltin(
    name = "JavaRuntimeInfo",
    category = StarlarkDocumentationCategory.PROVIDER,
    doc = "Information about the Java runtime being used.")
public interface JavaRuntimeInfoApi extends ToolchainInfoApi {

  @StarlarkMethod(
      name = "java_home",
      doc = "Returns the execpath of the root of the Java installation.",
      structField = true)
  String javaHome();

  /** The execpath of the Java binary. */
  @StarlarkMethod(
      name = "java_executable_exec_path",
      doc = "Returns the execpath of the Java executable.",
      structField = true)
  String javaBinaryExecPath();

  /** The runfiles path of the JDK. */
  @StarlarkMethod(
      name = "java_home_runfiles_path",
      doc =
          "Returns the path of the Java installation in runfiles trees. This should only be used "
              + "when one needs to access the JDK during the execution of a binary or a test built "
              + "by Bazel. In particular, when one needs the JDK during an action, "
              + "java_home should be used instead.",
      structField = true)
  String javaHomeRunfilesPath();

  /** The runfiles path of the Java binary. */
  @StarlarkMethod(
      name = "java_executable_runfiles_path",
      doc =
          "Returns the path of the Java executable in runfiles trees. This should only be used "
              + "when one needs to access the JVM during the execution of a binary or a test built "
              + "by Bazel. In particular, when one needs to invoke the JVM during an action, "
              + "java_executable_exec_path should be used instead.",
      structField = true)
  String javaBinaryRunfilesPath();

  /** The files in the Java runtime. */
  @StarlarkMethod(
      name = "files",
      doc = "Returns the files in the Java runtime.",
      structField = true)
  Depset starlarkJavaBaseInputs();
}
