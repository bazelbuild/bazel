// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.apple;

import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;

/** Provider structure for {@code XcodeVersionRuleData} */
interface XcodeVersionProviderApi<FileT extends FileApi> extends ProviderApi {

  @StarlarkMethod(
      name = "XcodeVersionRuleData",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "label",
            positional = false,
            named = true,
            defaultValue = "None",
            documented = false),
        @Param(
            name = "xcode_properties",
            positional = false,
            named = true,
            defaultValue = "None",
            documented = false),
        @Param(
            name = "aliases",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Sequence.class)},
            documented = false),
      },
      selfCall = true,
      documented = false)
  @StarlarkConstructor
  public XcodeVersionRuleData createInfo(
      Object starlarkLabel,
      Object starlarkXcodeProperties,
      Object starlarkAliases,
      StarlarkThread thread)
      throws EvalException;
}
