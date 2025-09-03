// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import javax.annotation.Nullable;

/**
 * Interface to mark classes that could contain transitive information added using the Starlark
 * framework.
 */
public interface ProviderCollection {
  /**
   * Returns the transitive information provider requested, or null if the provider is not found.
   * The provider has to be a TransitiveInfoProvider Java class.
   */
  @Nullable
  <P extends TransitiveInfoProvider> P getProvider(Class<P> provider);

  /**
   * Returns the transitive information requested or null, if the information is not found. The
   * transitive information has to have been added using the Starlark framework.
   */
  @Nullable
  Object get(String providerKey);

  /**
   * Returns the declared provider requested, or null, if the information is not found.
   *
   * <p>Use {@link #get(BuiltinProvider)} for built-in providers.
   */
  @Nullable
  Info get(Provider.Key providerKey);

  /**
   * Returns the native declared provider requested, or null, if the information is not found.
   *
   * <p>Type-safe version of {@link #get(Provider.Key)} for built-in providers.
   */
  @Nullable
  default <T extends Info> T get(BuiltinProvider<T> provider) {
    return provider.getValueClass().cast(get(provider.getKey()));
  }

  /**
   * Retrieves and converts an instance of a Starlark-defined provider to an instance of {@code T},
   * according to the conversion defined by {@code wrapper}.
   *
   * <p>If the provider identified by {@code wrapper} is not present, returns null.
   *
   * <p>Conversion errors (e.g. missing fields or bad types) are indicated by throwing {@link
   * RuleErrorException}.
   */
  @Nullable
  default <T> T get(StarlarkProviderWrapper<T> wrapper) throws RuleErrorException {
    Info value = get(wrapper.getKey());
    return value == null ? null : wrapper.wrap(value);
  }

  /**
   * Returns the provider defined in Starlark, or null, if the information is not found. The
   * transitive information has to have been added using the Starlark framework.
   *
   * <p>This method dispatches to either {@link #get(Provider.Key)} or {@link #get(String)}
   * depending on whether {@link StarlarkProviderIdentifier} is for legacy or for declared provider.
   */
  @Nullable
  default Object get(StarlarkProviderIdentifier id) {
    return this.get(id.getKey());
  }
}
