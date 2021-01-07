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

package com.google.devtools.build.lib.starlarkbuildapi.test;

import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/** Helper functions for Starlark to access coverage-related infrastructure */
@StarlarkBuiltin(
    name = "coverage_common",
    doc = "Helper functions to access coverage-related infrastructure.")
public interface CoverageCommonApi<
        ConstraintValueT extends ConstraintValueInfoApi,
        RuleContextT extends StarlarkRuleContextApi<ConstraintValueT>>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "instrumented_files_info",
      doc =
          "Creates a new "
              + "<a class=\"anchor\" href=\"InstrumentedFilesInfo.html\">InstrumentedFilesInfo</a> "
              + "instance. Use this provider to communicate coverage-related attributes of the "
              + "current build rule.",
      parameters = {
        @Param(name = "ctx", positional = true, named = true, doc = "The rule context."),
        @Param(
            name = "source_attributes",
            doc = "A list of attribute names which contain source files processed by this rule.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "dependency_attributes",
            doc =
                "A list of attribute names which might provide runtime dependencies (either code "
                    + "dependencies or runfiles).",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "extensions",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            doc =
                "File extensions used to filter files from source_attributes. For example, 'js'. "
                    + "If not provided (or None), then all files from source_attributes will be "
                    + "added to instrumented files, if an empty list is provided, then "
                    + "no files from source attributes will be added.",
            positional = false,
            named = true,
            defaultValue = "None"),
      })
  InstrumentedFilesInfoApi instrumentedFilesInfo(
      RuleContextT starlarkRuleContext,
      Sequence<?> sourceAttributes, // <String> expected
      Sequence<?> dependencyAttributes, // <String> expected
      Object extensions)
      throws EvalException;
}
