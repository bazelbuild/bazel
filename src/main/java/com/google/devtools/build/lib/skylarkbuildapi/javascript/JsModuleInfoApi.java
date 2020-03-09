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

package com.google.devtools.build.lib.skylarkbuildapi.javascript;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Info object propagating information about protocol buffer sources. */
@SkylarkModule(
    name = "JsModuleInfo",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Encapsulates information provided by js_module, a unit of JavaScript code that is part of"
            + " a bigger whole (a js_module_binary), but is not necessarily loaded at start-up.")
public interface JsModuleInfoApi<FileT extends FileApi> extends StructApi {
  /** Name of this info object. */
  String NAME = "JsModuleInfo";

  @SkylarkCallable(
      name = "label",
      doc = "Returns the label of the target which created this object",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  Label getLabel();

  @SkylarkCallable(
      name = "transitive_js_info",
      doc =
          "Returns the 'js' provider that contains information about this module and all of its"
              + " dependencies.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  StarlarkValue getFullPintoSources();

  /** Provider class for {@link JsModuleInfoApi} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  interface JsModuleInfoProviderApi extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>JsModuleInfo</code> constructor.",
        parameters = {
          @Param(
              name = "label",
              doc = "The label of the target which created this object",
              positional = false,
              named = true,
              type = Label.class),
          @Param(
              name = "wrapper",
              doc = "A string in which the output should be embedded.",
              positional = false,
              named = true,
              type = String.class),
          @Param(
              name = "full_pinto_sources",
              doc =
                  "PintoSourcesContextProvider for this module and the transitive closure of"
                      + " dependencies.",
              positional = false,
              named = true,
              type = Object.class),
          @Param(
              name = "direct_pinto_sources",
              doc = "PintoSourcesContextProvider for only this module.",
              positional = false,
              named = true,
              type = Object.class),
          @Param(
              name = "direct_module_dependencies",
              doc = "A list of direct module dependencies of this module.",
              positional = false,
              named = true,
              defaultValue = "[]",
              type = Sequence.class,
              generic1 = JsModuleInfoApi.class),
        },
        selfCall = true,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
    @SkylarkConstructor(objectType = JsModuleInfoApi.class, receiverNameForDoc = "JsModuleInfo")
    JsModuleInfoApi<?> jsModuleInfo(
        Label label,
        String wrapper,
        Object fullPintoSources,
        Object directPintoSources,
        Sequence<?> directModuleDependencies /* <? extends JsModuleApi> expected */)
        throws EvalException;
  }
}
