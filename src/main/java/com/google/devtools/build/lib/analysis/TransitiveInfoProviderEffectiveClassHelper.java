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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Provides the effective class for the provider. The effective class is inferred as the sole class
 * in the provider's inheritance hierarchy that implements {@link TransitiveInfoProvider} directly.
 * This allows for simple subclasses such as those created by AutoValue, but will fail if there's
 * any ambiguity as to which implementor of the {@link TransitiveInfoProvider} is intended. If the
 * provider implements multiple TransitiveInfoProvider interfaces, prefer the explicit put builder
 * methods.
 */
class TransitiveInfoProviderEffectiveClassHelper {

  private TransitiveInfoProviderEffectiveClassHelper() {}

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

  // TODO(arielb): see if these can be made private?
  static <T extends TransitiveInfoProvider> Class<T> get(T provider) {
    return get((Class<T>) provider.getClass());
  }

  static <T extends TransitiveInfoProvider> Class<T> get(Class<T> providerClass) {
    return (Class<T>) EFFECTIVE_PROVIDER_CLASS_CACHE.getUnchecked(providerClass);
  }
}
