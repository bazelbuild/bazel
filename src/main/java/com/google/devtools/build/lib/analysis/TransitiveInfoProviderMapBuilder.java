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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import java.util.Arrays;
import java.util.LinkedHashMap;
import javax.annotation.Nullable;

/** A builder for {@link TransitiveInfoProviderMap}. */
public class TransitiveInfoProviderMapBuilder {

  // TODO(arielb): share the instance with the outerclass and copy on write instead?
  private final LinkedHashMap<Object, Object> providers = new LinkedHashMap<>();

  /**
   * Returns <tt>true</tt> if a {@link TransitiveInfoProvider} has been added for the class
   * provided.
   */
  public boolean contains(Class<? extends TransitiveInfoProvider> providerClass) {
    return providers.containsKey(providerClass);
  }

  public boolean contains(String legacyId) {
    return providers.containsKey(legacyId);
  }

  public boolean contains(Provider.Key key) {
    return providers.containsKey(key);
  }


  public <T extends TransitiveInfoProvider> TransitiveInfoProviderMapBuilder put(
      Class<? extends T> providerClass, T provider) {
    Preconditions.checkNotNull(providerClass);
    Preconditions.checkNotNull(provider);
    Preconditions.checkState(
        !(provider instanceof Info), "Expose %s as native declared provider", providerClass);

    // TODO(arielb): throw an exception if the providerClass is already present?
    // This is enforced by aspects but RuleConfiguredTarget presents violations
    // particularly around LicensesProvider
    providers.put(providerClass, provider);
    return this;
  }

  public TransitiveInfoProviderMapBuilder put(Info classObject) {
    Preconditions.checkNotNull(classObject);
    Preconditions.checkState(!(classObject instanceof TransitiveInfoProvider),
        "Declared provider %s should not implement TransitiveInfoProvider",
        classObject.getClass());

    providers.put(classObject.getProvider().getKey(), classObject);
    return this;
  }

  public TransitiveInfoProviderMapBuilder put(String legacyKey, Object classObject) {
    Preconditions.checkNotNull(legacyKey);
    Preconditions.checkNotNull(classObject);
    providers.put(legacyKey, classObject);
    return this;
  }


  public TransitiveInfoProviderMapBuilder add(TransitiveInfoProvider provider) {
    return put(TransitiveInfoProviderEffectiveClassHelper.get(provider), provider);
  }

  public TransitiveInfoProviderMapBuilder add(TransitiveInfoProvider... providers) {
    return addAll(Arrays.asList(providers));
  }

  public TransitiveInfoProviderMapBuilder addAll(TransitiveInfoProviderMap other) {
    for (int i = 0; i < other.getProviderCount(); ++i) {
      providers.put(other.getProviderKeyAt(i), other.getProviderInstanceAt(i));
    }
    return this;
  }

  public TransitiveInfoProviderMapBuilder addAll(Iterable<TransitiveInfoProvider> providers) {
    for (TransitiveInfoProvider provider : providers) {
      add(provider);
    }
    return this;
  }

  @Nullable
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    return providerClass.cast(providers.get(providerClass));
  }

  @Nullable
  public Info getProvider(Provider.Key key) {
    return (Info) providers.get(key);
  }

  public TransitiveInfoProviderMap build() {
    return TransitiveInfoProviderMapImpl.create(providers);
  }
}
