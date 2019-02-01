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

import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainInfoApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Information about the Java runtime being used. */
@SkylarkModule(name = "JavaRuntimeInfo", doc = "Information about the Java runtime being used.")
public interface JavaRuntimeInfoApi extends ToolchainInfoApi {

  @SkylarkCallable(
      name = "java_home",
      doc = "Returns the execpath of the root of the Java installation.",
      structField = true
  )
  public PathFragment javaHome();

  /** The execpath of the Java binary. */
  @SkylarkCallable(
      name = "java_executable_exec_path",
      doc = "Returns the execpath of the Java executable.",
      structField = true)
  public PathFragment javaBinaryExecPath();

  /** The runfiles path of the JDK. */
  @SkylarkCallable(
      name = "java_home_runfiles_path",
      doc =
          "Returns the path of the Java installation in runfiles trees. This should only be used "
              + "when one needs to access the JDK during the execution of a binary or a test built "
              + "by Bazel. In particular, when one needs the JDK during an action, "
              + "java_home should be used instead.",
      structField = true)
  public PathFragment javaHomeRunfilesPath();

  /** The runfiles path of the Java binary. */
  @SkylarkCallable(
      name = "java_executable_runfiles_path",
      doc =
          "Returns the path of the Java executable in runfiles trees. This should only be used "
              + "when one needs to access the JVM during the execution of a binary or a test built "
              + "by Bazel. In particular, when one needs to invoke the JVM during an action, "
              + "java_executable_exec_path should be used instead.",
      structField = true)
  public PathFragment javaBinaryRunfilesPath();
}
