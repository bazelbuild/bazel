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

package com.google.devtools.build.lib.starlarkbuildapi.python;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Provider instance for {@code py_runtime}. */
@StarlarkBuiltin(
    name = "PyRuntimeInfo",
    doc =
        "Contains information about a Python runtime, as returned by the <code>py_runtime</code>"
            + "rule."
            + ""
            + "<p>A Python runtime describes either a <em>platform runtime</em> or an <em>in-build "
            + "runtime</em>. A platform runtime accesses a system-installed interpreter at a known "
            + "path, whereas an in-build runtime points to a <code>File</code> that acts as the "
            + "interpreter. In both cases, an \"interpreter\" is really any executable binary or "
            + "wrapper script that is capable of running a Python script passed on the command "
            + "line, following the same conventions as the standard CPython interpreter.",
    category = DocCategory.PROVIDER)
public interface PyRuntimeInfoApi<FileT extends FileApi> extends StarlarkValue {

  static final String DEFAULT_STUB_SHEBANG = "#!/usr/bin/env python3";

  @StarlarkMethod(
      name = "interpreter_path",
      structField = true,
      allowReturnNones = true,
      doc =
          "If this is a platform runtime, this field is the absolute filesystem path to the "
              + "interpreter on the target platform. Otherwise, this is <code>None</code>.")
  @Nullable
  String getInterpreterPathString();

  @StarlarkMethod(
      name = "interpreter",
      structField = true,
      allowReturnNones = true,
      doc =
          "If this is an in-build runtime, this field is a <code>File</code> representing the "
              + "interpreter. Otherwise, this is <code>None</code>. Note that an in-build runtime "
              + "can use either a prebuilt, checked-in interpreter or an interpreter built from "
              + "source.")
  @Nullable
  FileT getInterpreter();

  @StarlarkMethod(
      name = "files",
      structField = true,
      allowReturnNones = true,
      doc =
          "If this is an in-build runtime, this field is a <code>depset</code> of <code>File"
              + "</code>s that need to be added to the runfiles of an executable target that uses "
              + "this runtime (in particular, files needed by <code>interpreter</code>). The value "
              + "of <code>interpreter</code> need not be included in this "
              + "field. If this is a platform runtime then this field is <code>None</code>.")
  @Nullable
  Depset getFilesForStarlark();

  @StarlarkMethod(
      name = "python_version",
      structField = true,
      doc =
          "Indicates whether this runtime uses Python major version 2 or 3. Valid values are "
              + "(only) <code>\"PY2\"</code> and <code>\"PY3\"</code>.")
  String getPythonVersionForStarlark();

  @StarlarkMethod(
      name = "stub_shebang",
      structField = true,
      doc =
          "\"Shebang\" expression prepended to the bootstrapping Python stub script "
              + "used when executing <code>py_binary</code> targets.  Does not apply "
              + "to Windows.")
  String getStubShebang();

  /** Provider type for {@link PyRuntimeInfoApi} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface PyRuntimeInfoProviderApi extends ProviderApi {

    @StarlarkMethod(
        name = "PyRuntimeInfo",
        doc = "The <code>PyRuntimeInfo</code> constructor.",
        parameters = {
          @Param(
              name = "interpreter_path",
              allowedTypes = {
                @ParamType(type = String.class),
                @ParamType(type = NoneType.class),
              },
              positional = false,
              named = true,
              defaultValue = "None",
              doc =
                  "The value for the new object's <code>interpreter_path</code> field. Do not give "
                      + "a value for this argument if you pass in <code>interpreter</code>."),
          @Param(
              name = "interpreter",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              positional = false,
              named = true,
              defaultValue = "None",
              doc =
                  "The value for the new object's <code>interpreter</code> field. Do not give "
                      + "a value for this argument if you pass in <code>interpreter_path</code>."),
          @Param(
              name = "files",
              allowedTypes = {
                @ParamType(type = Depset.class, generic1 = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              positional = false,
              named = true,
              defaultValue = "None",
              doc =
                  "The value for the new object's <code>files</code> field. Do not give a value "
                      + "for this argument if you pass in <code>interpreter_path</code>. If "
                      + "<code>interpreter</code> is given and this argument is <code>None</code>, "
                      + "<code>files</code> becomes an empty <code>depset</code> instead."),
          @Param(
              name = "python_version",
              positional = false,
              named = true,
              doc = "The value for the new object's <code>python_version</code> field."),
          @Param(
              name = "stub_shebang",
              allowedTypes = {@ParamType(type = String.class)},
              positional = false,
              named = true,
              defaultValue = "'" + DEFAULT_STUB_SHEBANG + "'",
              doc =
                  "The value for the new object's <code>stub_shebang</code> field. "
                      + "Default is <code>"
                      + DEFAULT_STUB_SHEBANG
                      + "</code>."),
        },
        useStarlarkThread = true,
        selfCall = true)
    @StarlarkConstructor
    PyRuntimeInfoApi<?> constructor(
        Object interpreterPathUncast,
        Object interpreterUncast,
        Object filesUncast,
        String pythonVersion,
        String stubShebang,
        StarlarkThread thread)
        throws EvalException;
  }
}
