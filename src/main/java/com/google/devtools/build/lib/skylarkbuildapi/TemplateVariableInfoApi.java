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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/** Provides access to make variables from the current fragments. */
@SkylarkModule(
    name = "TemplateVariableInfo",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Encapsulates template variables, that is, variables that can be referenced by strings"
            + " like <code>$(VARIABLE)</code> in BUILD files and expanded by"
            + " <code>ctx.expand_make_variables</code> and implicitly in certain attributes of"
            + " built-in rules.</p><p><code>TemplateVariableInfo</code> can be created by calling"
            + " its eponymous constructor with a string-to-string dict as an argument that"
            + " specifies the variables provided.</p><p>Example:"
            + " <code>platform_common.TemplateVariableInfo({'FOO': 'bar'})</code></p>")
public interface TemplateVariableInfoApi extends StructApi {

  /** The global provider name. */
  String NAME = "TemplateVariableInfo";

  @SkylarkCallable(
      name = "variables",
      doc =
          "Returns the make variables defined by this target as a dictionary with string keys "
              + "and string values",
      structField = true)
  ImmutableMap<String, String> getVariables();

  /** Provider for {@link TemplateVariableInfoApi} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  interface Provider extends ProviderApi {

    @SkylarkCallable(
        name = "TemplateVariableInfo",
        doc = "The <code>TemplateVariableInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(name = "vars", positional = true, named = true, type = Dict.class),
        },
        selfCall = true,
        useStarlarkThread = true)
    TemplateVariableInfoApi templateVariableInfo(Dict<?, ?> vars, StarlarkThread thread)
        throws EvalException;
  }
}
