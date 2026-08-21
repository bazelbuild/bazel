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
        "A mutable copy of a target's declared providers. It is intentionally non-iterable; use"
            + " known provider constructors with index notation or the <code>in</code> operator"
            + " instead. Providers can be replaced or removed using the <code>add</code> and"
            + " <code>remove</code> methods. A ProviderMap is only usable during the rule or aspect"
            + " implementation function that created it. It may be returned directly from a rule"
            + " implementation function.")
public interface ProviderMapApi extends StarlarkValue, StarlarkIndexable {

  @StarlarkMethod(
      name = "add",
      doc = "Adds a provider instance, replacing any existing instance of the same provider.",
      parameters = {@Param(name = "provider", doc = "The provider instance to add.")})
  void add(Object provider) throws EvalException;

  @StarlarkMethod(
      name = "remove",
      doc = "Removes a provider. Fails if the provider is not present.",
      parameters = {@Param(name = "provider", doc = "The provider constructor to remove.")})
  void remove(Object provider) throws EvalException;
}
