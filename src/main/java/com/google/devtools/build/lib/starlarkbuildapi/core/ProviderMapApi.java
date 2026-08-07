// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.core;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkIndexable;
import net.starlark.java.eval.StarlarkValue;

/** A collection of providers returned by {@link TransitiveInfoCollectionApi}. */
@StarlarkBuiltin(
    name = "ProviderMap",
    category = DocCategory.BUILTIN,
    doc =
        "A non-iterable collection of providers. Providers can be accessed or assigned by type"
            + " using index notation, and their presence can be checked using the <code>in</code>"
            + " operator.")
public interface ProviderMapApi extends StarlarkValue, StarlarkIndexable.Settable {

  @StarlarkMethod(
      name = "pop",
      doc =
          "Removes a provider and returns its instance. If the provider is absent, returns the"
              + " specified <code>default</code> value, or fails if no default was specified.",
      parameters = {
        @Param(name = "provider", doc = "The provider constructor to remove."),
        @Param(
            name = "default",
            defaultValue = "unbound",
            named = true,
            doc = "The value to return if the provider is absent.")
      })
  Object pop(Object key, Object defaultValue) throws EvalException;
}
