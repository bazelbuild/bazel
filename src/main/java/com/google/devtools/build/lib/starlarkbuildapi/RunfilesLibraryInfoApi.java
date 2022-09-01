// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** Interface for runfiles library info. */
@StarlarkBuiltin(
    name = "RunfilesLibraryInfo",
    category = DocCategory.PROVIDER,
    doc =
        "Signals to direct dependents as well as Bazel itself that this target is a runfiles"
            + " library. For example, rules for compiled languages may choose to generate and"
            + " compile additional code that gives access to the name of the repository defining"
            + " the current target if they find this provider advertised by a direct"
            + " dependency.<p>This provider carries no data.")
public interface RunfilesLibraryInfoApi extends StructApi {

  /** Interface for runfiles library info provider. */
  @StarlarkBuiltin(name = "Provider", category = DocCategory.PROVIDER, documented = false)
  interface RunfilesLibraryInfoApiProvider extends ProviderApi {

    @StarlarkMethod(
        name = "RunfilesLibraryInfo",
        doc = "The <code>RunfilesLibraryInfo</code> constructor.",
        selfCall = true)
    @StarlarkConstructor
    RunfilesLibraryInfoApi constructor();
  }
}
