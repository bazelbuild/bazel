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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.StarlarkThread;

/** A provider that gives general information about a target's direct and transitive files. */
@StarlarkBuiltin(
    name = "DefaultInfo",
    category = DocCategory.PROVIDER,
    doc =
        "A provider that gives general information about a target's direct and transitive files."
            + " Every rule type has this provider, even if it is not returned explicitly by the"
            + " rule's implementation function. Each <code>DefaultInfo</code> instance has the"
            + " following fields: <ul><li><code>files</code><li><code>files_to_run</code>"
            + "<li><code>data_runfiles</code><li><code>default_runfiles</code></ul>See the <a"
            + " href='https://bazel.build/extending/rules'>rules</a> page for extensive guides on"
            + " how to use this provider.")
public interface DefaultInfoApi extends StructApi {

  static final String DEPRECATED_RUNFILES_PARAMETER_WARNING =
      "<p><b>It is recommended that you avoid using this parameter (see "
          + "<a href='https://bazel.build/extending/rules#runfiles_features_to_avoid'>"
          + "\"runfiles features to avoid\"</a>)</b></p> ";

  @StarlarkMethod(
      name = "files",
      doc =
          "A <a href='../builtins/depset.html'><code>depset</code></a> of <a"
              + " href='../builtins/File.html'><code>File</code></a> objects representing the"
              + " default outputs to build when this target is specified on the bazel command line."
              + " By default it is all predeclared outputs.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  Depset getFiles();

  @StarlarkMethod(
      name = "files_to_run",
      doc =
          "A <a href='../providers/FilesToRunProvider.html'><code>FilesToRunProvider</code></a>"
              + " object containing information about the executable and runfiles of the target.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FilesToRunProviderApi<?> getFilesToRun();

  @StarlarkMethod(
      name = "data_runfiles",
      doc =
          "runfiles descriptor describing the files that this target needs when run in the "
              + "condition that it is a <code>data</code> dependency attribute. Under most "
              + "circumstances, use the <code>default_runfiles</code> parameter instead. "
              + "See <a href='https://bazel.build/extending/rules#runfiles_features_to_avoid'>"
              + "\"runfiles features to avoid\"</a> for details. ",
      structField = true,
      allowReturnNones = true)
  @Nullable
  RunfilesApi getDataRunfiles();

  @StarlarkMethod(
      name = "default_runfiles",
      doc =
          "runfiles descriptor describing the files that this target needs when run "
              + "(via the <code>run</code> command or as a tool dependency).",
      structField = true,
      allowReturnNones = true)
  @Nullable
  RunfilesApi getDefaultRunfiles();

  /** Provider for {@link DefaultInfoApi}. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface DefaultInfoApiProvider<RunfilesT extends RunfilesApi, FileT extends FileApi>
      extends ProviderApi {

    @StarlarkMethod(
        name = "DefaultInfo",
        doc = "<p>The <code>DefaultInfo</code> constructor.",
        parameters = {
          @Param(
              name = "files",
              allowedTypes = {
                @ParamType(type = Depset.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              positional = false,
              defaultValue = "None",
              doc =
                  "A <a href='../builtins/depset.html'><code>depset</code></a> of <a"
                      + " href='../builtins/File.html'><code>File</code></a> objects representing"
                      + " the default outputs to build when this target is specified on the bazel"
                      + " command line. By default it is all predeclared outputs."),
          @Param(
              name = "runfiles",
              allowedTypes = {
                @ParamType(type = RunfilesApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              positional = false,
              defaultValue = "None",
              doc =
                  "runfiles descriptor describing the files that this target needs when run "
                      + "(via the <code>run</code> command or as a tool dependency)."),
          @Param(
              name = "data_runfiles",
              allowedTypes = {
                @ParamType(type = RunfilesApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              positional = false,
              defaultValue = "None",
              doc =
                  DEPRECATED_RUNFILES_PARAMETER_WARNING
                      + "runfiles descriptor describing the runfiles this target needs to run "
                      + "when it is a dependency via the <code>data</code> attribute."),
          @Param(
              name = "default_runfiles",
              allowedTypes = {
                @ParamType(type = RunfilesApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              positional = false,
              defaultValue = "None",
              doc =
                  DEPRECATED_RUNFILES_PARAMETER_WARNING
                      + "runfiles descriptor describing the runfiles this target needs to run "
                      + "when it is a dependency via any attribute other than the "
                      + "<code>data</code> attribute."),
          @Param(
              name = "executable",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              positional = false,
              defaultValue = "None",
              doc =
                  "If this rule is marked <a"
                      + " href='../globals/bzl.html#rule.executable'><code>executable</code></a> or"
                      + " <a href='../globals/bzl.html#rule.test'><code>test</code></a>, this is a"
                      + " <a href='../builtins/File.html'><code>File</code></a> object representing"
                      + " the file that should be executed to run the target. By default it is the"
                      + " predeclared output <code>ctx.outputs.executable</code>.")
        },
        selfCall = true,
        useStarlarkThread = true)
    @StarlarkConstructor
    DefaultInfoApi constructor(
        Object files,
        Object runfiles,
        Object dataRunfiles,
        Object defaultRunfiles,
        Object executable,
        StarlarkThread thread)
        throws EvalException;
  }
}
