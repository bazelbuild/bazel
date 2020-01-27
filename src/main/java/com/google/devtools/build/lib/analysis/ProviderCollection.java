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
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import javax.annotation.Nullable;

/**
 * Interface to mark classes that could contain transitive information added using the Skylark
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
   * transitive information has to have been added using the Skylark framework.
   */
  @Nullable
  Object get(String providerKey);

  /**
   * Returns the declared provider requested, or null, if the information is not found.
   *
   * <p>Use {@link #get(NativeProvider)} for native providers.
   */
  @Nullable
  Info get(Provider.Key providerKey);

  /**
   * Returns the native declared provider requested, or null, if the information is not found.
   *
   * <p>Type-safe version of {@link #get(Provider.Key)} for native providers.
   */
  @Nullable
  default <T extends Info> T get(NativeProvider<T> provider) {
    return provider.getValueClass().cast(get(provider.getKey()));
  }

  /**
   * Returns the native declared provider requested, or null, if the information is not found.
   *
   * <p>Type-safe version of {@link #get(Provider.Key)} for native providers.
   */
  @Nullable
  default <T extends Info> T get(BuiltinProvider<T> provider) {
    return provider.getValueClass().cast(get(provider.getKey()));
  }

  /**
   * Returns the provider defined in Skylark, or null, if the information is not found. The
   * transitive information has to have been added using the Skylark framework.
   *
   * <p>This method dispatches to either {@link #get(Provider.Key)} or {@link #get(String)}
   * depending on whether {@link SkylarkProviderIdentifier} is for legacy or for declared provider.
   */
  @Nullable
  default Object get(SkylarkProviderIdentifier id) {
    if (id.isLegacy()) {
      return this.get(id.getLegacyId());
    } else {
      return this.get(id.getKey());
    }
  }
}
