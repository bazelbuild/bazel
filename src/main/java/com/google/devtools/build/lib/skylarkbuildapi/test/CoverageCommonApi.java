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

package com.google.devtools.build.lib.skylarkbuildapi.test;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;

/** Helper functions for Starlark to access coverage-related infrastructure */
@SkylarkModule(
    name = "coverage_common",
    doc = "Helper functions for Starlark to access coverage-related infrastructure.")
public interface CoverageCommonApi<RuleContextT extends SkylarkRuleContextApi>
    extends SkylarkValue {

  @SkylarkCallable(
      name = "instrumented_files_info",
      doc =
          "Creates a new execution info provider. Use this provider to specify special"
              + "environments requirements needed to run tests.",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "source_attributes",
            doc = "A list of attribute names which contain source files for this rule.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "dependency_attributes",
            doc =
                "A list of attribute names which contain dependencies that might include "
                    + "instrumented files.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "extensions",
            doc =
                "File extensions used to filter files from source_attributes. For example, 'js'. "
                    + "If not provided (or None), then all files from source_attributes will be "
                    + "added to instrumented files, if an empty list is provided, then "
                    + "no files from source attributes will be added.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            type = SkylarkList.class),
      },
      useLocation = true)
  public InstrumentedFilesInfoApi instrumentedFilesInfo(
      RuleContextT skylarkRuleContext,
      SkylarkList<String> sourceAttributes,
      SkylarkList<String> dependencyAttributes,
      Object extensions,
      Location location)
      throws EvalException;
}
