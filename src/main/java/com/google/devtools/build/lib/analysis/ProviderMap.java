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

import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderMapApi;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A mutable, non-iterable Starlark map from declared provider constructors to provider instances.
 *
 * <p>A map exposed to Starlark is scoped to the rule or aspect implementation that created it. It
 * may be returned directly as a rule's provider collection.
 */
public final class ProviderMap implements ProviderMapApi, Mutability.Freezable {
  private final LinkedHashMap<Provider.Key, Info> providers;
  private final Mutability mutability;

  /**
   * Returns a new map that takes ownership of {@code providers}. The caller must not subsequently
   * mutate the supplied map.
   */
  public static ProviderMap create(LinkedHashMap<Provider.Key, Info> providers) {
    return new ProviderMap(providers, Mutability.IMMUTABLE);
  }

  private ProviderMap(LinkedHashMap<Provider.Key, Info> providers, Mutability mutability) {
    this.providers = providers;
    this.mutability = mutability;
  }

  ProviderMap mutableCopy(Mutability mutability) {
    return new ProviderMap(new LinkedHashMap<>(providers), mutability);
  }

  /** Returns the provider instances for consumption by rule implementation machinery. */
  public Collection<Info> getProviderInstances() {
    return Collections.unmodifiableCollection(providers.values());
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
    checkUsable();
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
    checkUsable();
    return providers.containsKey(selectExportedProvider(key, "querying").getKey());
  }

  @Override
  public void add(Object value) throws EvalException {
    checkUsable();
    Starlark.checkMutable(this);
    if (!(value instanceof Info info)) {
      throw Starlark.errorf(
          "ProviderMap.add() requires a provider instance, got %s", Starlark.type(value));
    }
    Provider constructor = info.getProvider();
    if (!constructor.isExported()) {
      throw Starlark.errorf(
          "ProviderMap only accepts instances of exported providers. Assign the provider a name "
              + "in a top-level assignment statement.");
    }
    providers.put(constructor.getKey(), info);
  }

  @Override
  public void remove(Object key) throws EvalException {
    checkUsable();
    Starlark.checkMutable(this);
    Provider constructor = selectExportedProvider(key, "removing");
    if (providers.remove(constructor.getKey()) == null) {
      throw Starlark.errorf(
          "ProviderMap doesn't contain declared provider '%s'", constructor.getPrintableName());
    }
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("<ProviderMap");
    String separator = ": ";
    for (Info info : providers.values()) {
      printer.append(separator).append(info.getProvider().getPrintableName());
      separator = ", ";
    }
    printer.append(">");
  }

  private void checkUsable() throws EvalException {
    if (mutability.isFrozen()) {
      throw Starlark.errorf(
          "cannot access ProviderMap outside of its owning rule or aspect implementation"
              + " function");
    }
  }

  private static Provider selectExportedProvider(Object key, String operation)
      throws EvalException {
    if (!(key instanceof Provider constructor)) {
      throw Starlark.errorf(
          "Type ProviderMap only supports %s by provider constructors, got %s instead",
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
