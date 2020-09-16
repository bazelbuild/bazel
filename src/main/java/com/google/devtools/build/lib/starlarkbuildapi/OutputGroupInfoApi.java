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

import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkConstructor;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;

/** Interface for an info object that indicates what output groups a rule has. */
@StarlarkBuiltin(
    name = "OutputGroupInfo",
    category = StarlarkDocumentationCategory.PROVIDER,
    doc =
        "A provider that indicates what output groups a rule has.<br>"
            + "See <a href=\"../rules.$DOC_EXT#requesting-output-files\">Requesting output files"
            + "</a> for more information.")
public interface OutputGroupInfoApi extends StructApi {

  /** Provider for {@link OutputGroupInfoApi}. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface OutputGroupInfoApiProvider extends ProviderApi {

    @StarlarkMethod(
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
        selfCall = true)
    @StarlarkConstructor(
        objectType = OutputGroupInfoApi.class,
        receiverNameForDoc = "OutputGroupInfo")
    OutputGroupInfoApi constructor(Dict<String, Object> kwargs) throws EvalException;
  }
}
