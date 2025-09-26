// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/**
 * Context object that is passed to the ctx.actions.map_directory implementation to allow for the
 * creation of actions.
 */
@StarlarkBuiltin(
    name = "template_ctx",
    category = DocCategory.BUILTIN,
    doc = "A context object that is passed to the action template expansion function.")
public interface StarlarkTemplateContextApi extends StarlarkValue {

  @StarlarkMethod(
      name = "declare_file",
      doc =
          "Declares that implementation creates a file with the given filename within the specified"
              + " directory.<p>Remember that in addition to declaring a file, you must separately"
              + " create an action that emits the file. Creating that action will require passing"
              + " the returned <code>File</code> object to the action's construction"
              + " function.",
      parameters = {
        @Param(name = "filename", doc = "The relative path of the file within the directory."),
        @Param(
            name = "directory",
            doc = "The directory in which the file should be created.",
            allowedTypes = {
              @ParamType(type = FileApi.class),
            },
            named = true,
            positional = false)
      })
  FileApi declareFile(String filename, FileApi directory) throws EvalException;

  @StarlarkMethod(
      name = "args",
      doc = "Returns an Args object that can be used to build memory-efficient command lines.",
      useStarlarkThread = true)
  CommandLineArgsApi args(StarlarkThread thread);

  @StarlarkMethod(
      name = "run",
      doc = "Creates an action that runs an executable.",
      parameters = {
        @Param(
            name = "outputs",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = FileApi.class),
            },
            named = true,
            positional = false,
            doc = "List of the output files of the action."),
        @Param(
            name = "inputs",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = FileApi.class),
              @ParamType(type = Depset.class),
            },
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = "List or depset of the input files of the action."),
        @Param(
            name = "executable",
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = String.class),
              @ParamType(type = FilesToRunProviderApi.class),
            },
            named = true,
            positional = false,
            doc = "The executable file to be called by the action."),
        @Param(
            name = "tools",
            allowedTypes = {
              @ParamType(type = Sequence.class),
              @ParamType(type = Depset.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc = StarlarkActionFactoryApi.TOOLS_ARG_DOC),
        @Param(
            name = "arguments",
            allowedTypes = {@ParamType(type = Sequence.class)},
            defaultValue = "[]",
            named = true,
            positional = false,
            doc =
                "Command line arguments of the action. "
                    + "Must be a list of strings or "
                    + "<a href=\"#args\"><code>actions.args()</code></a> objects."),
        @Param(
            name = "progress_message",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc = "Progress message to show to the user during the build."),
      })
  void run(
      Sequence<?> outputs,
      Object inputs,
      Object executableUnchecked,
      Object toolsUnchecked,
      Sequence<?> arguments,
      Object progressMessage)
      throws EvalException, InterruptedException;
}
