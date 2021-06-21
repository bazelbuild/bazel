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

import static com.google.common.base.Preconditions.checkState;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.collect.Sets;
import java.util.Set;

/**
 * Provides the effective class for the provider. The effective class is inferred as the sole class
 * in the provider's inheritance hierarchy that implements {@link TransitiveInfoProvider} directly.
 * This allows for simple subclasses such as those created by AutoValue, but will fail if there's
 * any ambiguity as to which implementor of the {@link TransitiveInfoProvider} is intended. If the
 * provider implements multiple TransitiveInfoProvider interfaces, prefer the explicit put builder
 * methods.
 */
final class TransitiveInfoProviderEffectiveClassHelper {

  private TransitiveInfoProviderEffectiveClassHelper() {}

  private static final LoadingCache<
          Class<? extends TransitiveInfoProvider>, Class<? extends TransitiveInfoProvider>>
      effectiveProviderClassCache =
          Caffeine.newBuilder()
              .build(TransitiveInfoProviderEffectiveClassHelper::findEffectiveProviderClass);

  private static Class<? extends TransitiveInfoProvider> findEffectiveProviderClass(
      Class<? extends TransitiveInfoProvider> providerClass) {
    Set<Class<? extends TransitiveInfoProvider>> result = getDirectImplementations(providerClass);
    checkState(
        result.size() == 1,
        "Effective provider class for %s is ambiguous (%s), specify explicitly.",
        providerClass,
        result);
    return result.iterator().next();
  }

  private static Set<Class<? extends TransitiveInfoProvider>> getDirectImplementations(
      Class<? extends TransitiveInfoProvider> providerClass) {
    Set<Class<? extends TransitiveInfoProvider>> result = Sets.newLinkedHashSetWithExpectedSize(1);
    for (Class<?> clazz : providerClass.getInterfaces()) {
      if (TransitiveInfoProvider.class.equals(clazz)) {
        result.add(providerClass);
      } else if (TransitiveInfoProvider.class.isAssignableFrom(clazz)) {
        result.addAll(getDirectImplementations(clazz.asSubclass(TransitiveInfoProvider.class)));
      }
    }

    Class<?> superclass = providerClass.getSuperclass();
    if (superclass != null && TransitiveInfoProvider.class.isAssignableFrom(superclass)) {
      result.addAll(getDirectImplementations(superclass.asSubclass(TransitiveInfoProvider.class)));
    }
    return result;
  }

  @SuppressWarnings("unchecked")
  static <T extends TransitiveInfoProvider> Class<T> get(T provider) {
    return get((Class<T>) provider.getClass());
  }

  @SuppressWarnings("unchecked")
  static <T extends TransitiveInfoProvider> Class<T> get(Class<T> providerClass) {
    return (Class<T>) effectiveProviderClassCache.get(providerClass);
  }
}
