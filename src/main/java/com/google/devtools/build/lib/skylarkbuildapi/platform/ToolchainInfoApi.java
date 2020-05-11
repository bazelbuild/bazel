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

package com.google.devtools.build.lib.skylarkbuildapi.platform;

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDocumentationCategory;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/** Info object representing data about a specific toolchain. */
@StarlarkBuiltin(
    name = "ToolchainInfo",
    doc =
        "Provider returned by <a href=\"../../toolchains.html#defining-toolchains\">toolchain "
            + "rules</a> to share data with "
            + "<a href=\"../../toolchains.html#writing-rules-that-use-toolchains\">rules which "
            + "depend on toolchains</a>. Read about <a href='../../toolchains.html'>"
            + "toolchains</a> for more information.",
    category = StarlarkDocumentationCategory.PROVIDER)
public interface ToolchainInfoApi extends StructApi {

  /** Provider for {@link ToolchainInfoApi} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface Provider extends ProviderApi {

    @SkylarkCallable(
        name = "ToolchainInfo",
        doc = "The <code>ToolchainInfo</code> constructor.",
        documented = false,
        extraKeywords = @Param(name = "kwargs", doc = "Dictionary of additional entries."),
        selfCall = true,
        useStarlarkThread = true)
    ToolchainInfoApi toolchainInfo(Dict<String, Object> kwargs, StarlarkThread thread)
        throws EvalException;
  }
}
