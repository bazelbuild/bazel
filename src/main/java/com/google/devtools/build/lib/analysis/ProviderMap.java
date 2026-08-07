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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderMapApi;
import java.util.LinkedHashMap;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/** A non-iterable Starlark map from declared provider constructors to provider instances. */
public final class ProviderMap implements ProviderMapApi, Mutability.Freezable {
  private final LinkedHashMap<Provider.Key, Info> providers;
  private final Mutability mutability;

  /** Returns a new map containing {@code providers}. */
  public static ProviderMap create(Iterable<Info> providers) {
    LinkedHashMap<Provider.Key, Info> providersByKey = new LinkedHashMap<>();
    for (Info provider : providers) {
      providersByKey.put(provider.getProvider().getKey(), provider);
    }
    return new ProviderMap(providersByKey, Mutability.IMMUTABLE);
  }

  /** Returns a new empty map. */
  public static ProviderMap empty() {
    return new ProviderMap(new LinkedHashMap<>(), Mutability.IMMUTABLE);
  }

  private ProviderMap(LinkedHashMap<Provider.Key, Info> providers, Mutability mutability) {
    this.providers = providers;
    this.mutability = mutability;
  }

  ProviderMap mutableCopy(Mutability mutability) {
    return new ProviderMap(new LinkedHashMap<>(providers), mutability);
  }

  /** Returns the provider instances for consumption by rule implementation machinery. */
  public ImmutableCollection<Info> getProviderInstances() {
    return ImmutableList.copyOf(providers.values());
  }

  @Override
  public Mutability mutability() {
    return mutability;
  }

  @Override
  public boolean isImmutable() {
    return mutability.isFrozen();
  }

  @Override
  public void checkHashable() throws EvalException {
    throw Starlark.errorf("unhashable type: 'ProviderMap'");
  }

  @Override
  public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    Provider constructor = selectExportedProvider(key, "indexing");
    Info provider = providers.get(constructor.getKey());
    if (provider != null) {
      return provider;
    }
    throw Starlark.errorf(
        "ProviderMap doesn't contain declared provider '%s'", constructor.getPrintableName());
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    return providers.containsKey(selectExportedProvider(key, "querying").getKey());
  }

  @Override
  public void setIndex(StarlarkSemantics semantics, Object key, Object value) throws EvalException {
    Starlark.checkMutable(this);
    Provider constructor = selectExportedProvider(key, "assigning");
    if (!(value instanceof Info info)) {
      throw Starlark.errorf(
          "ProviderMap values must be provider instances, got %s", Starlark.type(value));
    }
    Provider valueConstructor = info.getProvider();
    if (!valueConstructor.isExported()) {
      throw Starlark.errorf(
          "ProviderMap only accepts instances of exported providers. Assign the provider a name "
              + "in a top-level assignment statement.");
    }
    if (!constructor.getKey().equals(valueConstructor.getKey())) {
      throw Starlark.errorf(
          "cannot assign an instance of provider '%s' to ProviderMap key '%s'",
          valueConstructor.getPrintableName(), constructor.getPrintableName());
    }
    providers.put(constructor.getKey(), info);
  }

  @Override
  public Object pop(Object key, Object defaultValue) throws EvalException {
    Starlark.checkMutable(this);
    Provider constructor = selectExportedProvider(key, "popping");
    Info provider = providers.remove(constructor.getKey());
    if (provider != null) {
      return provider;
    }
    if (defaultValue != Starlark.UNBOUND) {
      return defaultValue;
    }
    throw Starlark.errorf(
        "ProviderMap doesn't contain declared provider '%s'", constructor.getPrintableName());
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("<provider map>");
  }

  private static Provider selectExportedProvider(Object key, String operation)
      throws EvalException {
    if (!(key instanceof Provider constructor)) {
      throw Starlark.errorf(
          "Type ProviderMap only supports %s by object constructors, got %s instead",
          operation, Starlark.type(key));
    }
    if (!constructor.isExported()) {
      throw Starlark.errorf(
          "ProviderMap only supports %s by exported providers. Assign the provider a name "
              + "in a top-level assignment statement.",
          operation);
    }
    return constructor;
  }
}
