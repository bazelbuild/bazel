// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** The provider returned from Materialize Rules to materialize dependencies. */
@StarlarkBuiltin(
    name = "MaterializedDepsInfo",
    category = DocCategory.PROVIDER,
    doc = "The provider returned from materializer rules to materialize dependencies.")
public interface MaterializedDepsInfoApi extends StructApi {

  /** The global provider name. */
  String NAME = "MaterializedDepsInfo";

  @StarlarkMethod(
      name = "deps",
      doc = "The list of dependencies. These may be ConfiguredTarget or DormantDependency objects.",
      structField = true)
  ImmutableList<?> getDeps();

  /** Provider for {@link MaterializedDepsInfoApi} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface Provider extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        selfCall = true,
        doc = "The <code>MaterializedDepsInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "deps",
              positional = true,
              named = true,
              allowedTypes = {@ParamType(type = Sequence.class)})
        })
    @StarlarkConstructor
    MaterializedDepsInfoApi materializedDepsInfo(Sequence<?> dependencies) throws EvalException;
  }
}
