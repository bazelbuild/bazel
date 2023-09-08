// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/** Provider which describes a set of transitive package specifications used in package groups. */
@StarlarkBuiltin(
    name = "PackageSpecificationInfo",
    doc = "Information about transitive package specifications used in package groups.",
    category = DocCategory.PROVIDER)
public interface PackageSpecificationProviderApi extends StructApi {
  @StarlarkMethod(
      name = "contains",
      doc = "Checks if a target exists in a package group.",
      parameters = {
        @Param(
            name = "target",
            positional = true,
            doc = "A target which is checked if it exists inside the package group.",
            allowedTypes = {@ParamType(type = Label.class), @ParamType(type = String.class)})
      })
  public boolean targetInAllowlist(Object target) throws EvalException, LabelSyntaxException;
}
