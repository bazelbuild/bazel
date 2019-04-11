// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkList;

/** A collection of global skylark build API functions that apply to WORKSPACE files. */
@SkylarkGlobalLibrary
public interface WorkspaceGlobalsApi {

  @SkylarkCallable(
      name = "workspace",
      doc =
          "Sets the name for this workspace. Workspace names should be a Java-package-style "
              + "description of the project, using underscores as separators, e.g., "
              + "github.com/bazelbuild/bazel should use com_github_bazelbuild_bazel. Names must "
              + "start with a letter and can only contain letters, numbers, and underscores.",
      allowReturnNones = true,
      parameters = {
        @Param(
            name = "name",
            type = String.class,
            doc = "the name of the workspace.",
            named = true,
            positional = false)
      },
      useAst = true,
      useEnvironment = true)
  NoneType workspace(String name, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "register_execution_platforms",
      doc = "Registers a platform so that it is available to execute actions.",
      allowReturnNones = true,
      extraPositionals =
          @Param(
              name = "platform_labels",
              type = SkylarkList.class,
              generic1 = String.class,
              doc = "The labels of the platforms to register."),
      useLocation = true,
      useEnvironment = true)
  NoneType registerExecutionPlatforms(
      SkylarkList<?> platformLabels, Location location, Environment env)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "register_toolchains",
      doc =
          "Registers a toolchain created with the toolchain() rule so that it is available for "
              + "toolchain resolution.",
      allowReturnNones = true,
      extraPositionals =
          @Param(
              name = "toolchain_labels",
              type = SkylarkList.class,
              generic1 = String.class,
              doc = "The labels of the toolchains to register."),
      useLocation = true,
      useEnvironment = true)
  NoneType registerToolchains(SkylarkList<?> toolchainLabels, Location location, Environment env)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "bind",
      doc =
          "<p>Warning: use of <code>bind()</code> is not recommended. See <a"
              + " href=\"https://github.com/bazelbuild/bazel/issues/1952\">Consider removing"
              + " bind</a> for a long discussion if its issues and alternatives.</p> <p>Gives a"
              + " target an alias in the <code>//external</code> package.</p>",
      allowReturnNones = true,
      parameters = {
        @Param(
            name = "name",
            type = String.class,
            named = true,
            positional = false,
            doc = "The label under '//external' to serve as the alias name"),
        @Param(
            name = "actual",
            type = String.class,
            named = true,
            positional = false,
            noneable = true,
            defaultValue = "None",
            doc = "The real label to be aliased")
      },
      useAst = true,
      useEnvironment = true)
  NoneType bind(String name, Object actual, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException;
}
