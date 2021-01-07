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

package com.google.devtools.build.lib.starlarkbuildapi.javascript;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/** Info object propagating information about protocol buffer sources. */
@StarlarkBuiltin(
    name = "JsModuleInfo",
    category = DocCategory.PROVIDER,
    doc =
        "Encapsulates information provided by js_module, a unit of JavaScript code that is part of"
            + " a bigger whole (a js_module_binary), but is not necessarily loaded at start-up.")
public interface JsModuleInfoApi<FileT extends FileApi> extends StructApi {
  /** Name of this info object. */
  String NAME = "JsModuleInfo";

  @StarlarkMethod(
      name = "label",
      doc = "Returns the label of the target which created this object",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  Label getLabel();

  @StarlarkMethod(
      name = "transitive_js_info",
      doc =
          "Returns the 'js' provider that contains information about this module and all of its"
              + " dependencies.",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  StarlarkValue getFullPintoSources();

  /** Provider class for {@link JsModuleInfoApi} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface JsModuleInfoProviderApi extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>JsModuleInfo</code> constructor.",
        parameters = {
          @Param(
              name = "label",
              doc = "The label of the target which created this object",
              positional = false,
              named = true),
          @Param(
              name = "wrapper",
              doc = "A string in which the output should be embedded.",
              positional = false,
              named = true),
          @Param(
              name = "full_pinto_sources",
              doc =
                  "PintoSourcesContextProvider for this module and the transitive closure of"
                      + " dependencies.",
              positional = false,
              named = true),
          @Param(
              name = "direct_pinto_sources",
              doc = "PintoSourcesContextProvider for only this module.",
              positional = false,
              named = true),
          @Param(
              name = "direct_module_dependencies",
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = JsModuleInfoApi.class),
              },
              doc = "A list of direct module dependencies of this module.",
              positional = false,
              named = true,
              defaultValue = "[]"),
        },
        selfCall = true,
        enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
    @StarlarkConstructor
    JsModuleInfoApi<?> jsModuleInfo(
        Label label,
        String wrapper,
        Object fullPintoSources,
        Object directPintoSources,
        Sequence<?> directModuleDependencies /* <? extends JsModuleApi> expected */)
        throws EvalException;
  }
}
