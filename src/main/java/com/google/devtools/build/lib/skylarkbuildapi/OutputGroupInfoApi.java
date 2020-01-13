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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * Interface for an info object that indicates what output groups a rule has.
 */
@SkylarkModule(
    name = "OutputGroupInfo",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "A provider that indicates what output groups a rule has.<br>"
        + "See <a href=\"../rules.$DOC_EXT#requesting-output-files\">Requesting output files"
        + "</a> for more information."
)
public interface OutputGroupInfoApi extends StructApi {

  /** Provider for {@link OutputGroupInfoApi}. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  interface OutputGroupInfoApiProvider extends ProviderApi {

    @SkylarkCallable(
        name = "OutputGroupInfo",
        doc =
            "Instantiate this provider with <br>"
                + "<pre class=language-python>"
                + "OutputGroupInfo(group1 = &lt;files&gt;, group2 = &lt;files&gt;...)</pre>"
                + "See <a href=\"../rules.$DOC_EXT#requesting-output-files\">Requesting output "
                + "files </a> for more information.",
        extraKeywords =
            @Param(
                name = "kwargs",
                type = Dict.class,
                defaultValue = "{}",
                doc = "Dictionary of arguments."),
        useLocation = true,
        selfCall = true)
    @SkylarkConstructor(
        objectType = OutputGroupInfoApi.class,
        receiverNameForDoc = "OutputGroupInfo")
    OutputGroupInfoApi constructor(Dict<?, ?> kwargs, Location loc) throws EvalException;
  }
}
