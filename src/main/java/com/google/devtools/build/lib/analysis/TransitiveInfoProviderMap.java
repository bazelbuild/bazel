// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.Verify;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/** Provides a mapping between a TransitiveInfoProvider class and an instance. */
@Immutable
public final class TransitiveInfoProviderMap {

  private final ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> map;

  private TransitiveInfoProviderMap(
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> map) {
    this.map = map;
  }

  /** Initializes a {@link TransitiveInfoProviderMap} from the instances provided. */
  public static TransitiveInfoProviderMap of(TransitiveInfoProvider... providers) {
    return builder().add(providers).build();
  }

  /** Returns the instance for the provided providerClass, or <tt>null</tt> if not present. */
  @Nullable
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    return (P) map.get(getEffectiveProviderClass(providerClass));
  }

  public ImmutableSet<Map.Entry<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>>
      entrySet() {
    return map.entrySet();
  }

  public ImmutableCollection<TransitiveInfoProvider> values() {
    return map.values();
  }

  public Builder toBuilder() {
    return builder().addAll(map.values());
  }

  public static Builder builder() {
    return new Builder();
  }

  /** A builder for {@link TransitiveInfoProviderMap}. */
  public static class Builder {

    // TODO(arielb): share the instance with the outerclass and copy on write instead?
    private final LinkedHashMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
        providers = new LinkedHashMap();

    /**
     * Returns <tt>true</tt> if a {@link TransitiveInfoProvider} has been added for the class
     * provided.
     */
    public boolean contains(Class<? extends TransitiveInfoProvider> providerClass) {
      return providers.containsKey(providerClass);
    }

    public <T extends TransitiveInfoProvider> Builder put(
        Class<? extends T> providerClass, T provider) {
      Preconditions.checkNotNull(providerClass);
      Preconditions.checkNotNull(provider);
      // TODO(arielb): throw an exception if the providerClass is already present?
      // This is enforced by aspects but RuleConfiguredTarget presents violations
      // particularly around LicensesProvider
      providers.put(providerClass, provider);
      return this;
    }

    public Builder add(TransitiveInfoProvider provider) {
      return put(getEffectiveProviderClass(provider), provider);
    }

    public Builder add(TransitiveInfoProvider... providers) {
      return addAll(Arrays.asList(providers));
    }

    public Builder addAll(TransitiveInfoProviderMap providers) {
      return addAll(providers.values());
    }

    public Builder addAll(Iterable<TransitiveInfoProvider> providers) {
      for (TransitiveInfoProvider provider : providers) {
        add(provider);
      }
      return this;
    }

    @Nullable
    public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
      return (P) providers.get(providerClass);
    }

    public TransitiveInfoProviderMap build() {
      return new TransitiveInfoProviderMap(ImmutableMap.copyOf(providers));
    }
  }

  private static final LoadingCache<
          Class<? extends TransitiveInfoProvider>, Class<? extends TransitiveInfoProvider>>
      EFFECTIVE_PROVIDER_CLASS_CACHE =
          CacheBuilder.newBuilder()
              .build(
                  new CacheLoader<
                      Class<? extends TransitiveInfoProvider>,
                      Class<? extends TransitiveInfoProvider>>() {

                    private Set<Class<? extends TransitiveInfoProvider>> getDirectImplementations(
                        Class<? extends TransitiveInfoProvider> providerClass) {
                      Set<Class<? extends TransitiveInfoProvider>> result = new LinkedHashSet<>();
                      for (Class<?> clazz : providerClass.getInterfaces()) {
                        if (TransitiveInfoProvider.class.equals(clazz)) {
                          result.add(providerClass);
                        } else if (TransitiveInfoProvider.class.isAssignableFrom(clazz)) {
                          result.addAll(
                              getDirectImplementations(
                                  (Class<? extends TransitiveInfoProvider>) clazz));
                        }
                      }

                      Class<?> superclass = providerClass.getSuperclass();
                      if (superclass != null
                          && TransitiveInfoProvider.class.isAssignableFrom(superclass)) {
                        result.addAll(
                            getDirectImplementations(
                                (Class<? extends TransitiveInfoProvider>) superclass));
                      }
                      return result;
                    }

                    @Override
                    public Class<? extends TransitiveInfoProvider> load(
                        Class<? extends TransitiveInfoProvider> providerClass) {
                      Set<Class<? extends TransitiveInfoProvider>> result =
                          getDirectImplementations(providerClass);
                      Verify.verify(!result.isEmpty()); // impossible
                      Preconditions.checkState(
                          result.size() == 1,
                          "Effective provider class for %s is ambiguous (%s), specify explicitly.",
                          providerClass,
                          Joiner.on(',').join(result));
                      return result.iterator().next();
                    }
                  });

  /**
   * Provides the effective class for the provider. The effective class is inferred as the sole
   * class in the provider's inheritence hierarchy that implements {@link TransitiveInfoProvider}
   * directly. This allows for simple subclasses such as those created by AutoValue, but will fail
   * if there's any ambiguity as to which implementor of the {@link TransitiveInfoProvider} is
   * intended. If the provider implements multiple TransitiveInfoProvider interfaces, prefer the
   * explicit put builder methods.
   */
  // TODO(arielb): see if these can be made private?
  static <T extends TransitiveInfoProvider> Class<T> getEffectiveProviderClass(T provider) {
    return getEffectiveProviderClass((Class<T>) provider.getClass());
  }

  private static <T extends TransitiveInfoProvider> Class<T> getEffectiveProviderClass(
      Class<T> providerClass) {
    return (Class<T>) EFFECTIVE_PROVIDER_CLASS_CACHE.getUnchecked(providerClass);
  }
}
