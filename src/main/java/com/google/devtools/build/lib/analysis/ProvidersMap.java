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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProvidersMapApi;
import java.util.LinkedHashMap;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/** A non-iterable Starlark map from declared provider constructors to provider instances. */
public final class ProvidersMap implements ProvidersMapApi {
  private final ImmutableMap<Provider.Key, Info> providers;

  /** Returns a new map containing {@code providers}. */
  public static ProvidersMap create(Iterable<Info> providers) {
    LinkedHashMap<Provider.Key, Info> providersByKey = new LinkedHashMap<>();
    for (Info provider : providers) {
      providersByKey.put(provider.getProvider().getKey(), provider);
    }
    return new ProvidersMap(ImmutableMap.copyOf(providersByKey));
  }

  /** Returns a new empty map. */
  public static ProvidersMap empty() {
    return new ProvidersMap(ImmutableMap.of());
  }

  private ProvidersMap(ImmutableMap<Provider.Key, Info> providers) {
    this.providers = providers;
  }

  /** Returns the provider instances for consumption by rule implementation machinery. */
  public ImmutableCollection<Info> getProviderInstances() {
    return providers.values();
  }

  @Override
  public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    Provider constructor = selectExportedProvider(key, "index");
    Info provider = providers.get(constructor.getKey());
    if (provider != null) {
      return provider;
    }
    throw Starlark.errorf(
        "ProvidersMap doesn't contain declared provider '%s'", constructor.getPrintableName());
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    return providers.containsKey(selectExportedProvider(key, "query").getKey());
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("<providers map>");
  }

  private static Provider selectExportedProvider(Object key, String operation)
      throws EvalException {
    if (!(key instanceof Provider constructor)) {
      throw Starlark.errorf(
          "Type ProvidersMap only supports %sing by object constructors, got %s instead",
          operation, Starlark.type(key));
    }
    if (!constructor.isExported()) {
      throw Starlark.errorf(
          "ProvidersMap only supports %sing by exported providers. Assign the provider a name "
              + "in a top-level assignment statement.",
          operation);
    }
    return constructor;
  }
}
