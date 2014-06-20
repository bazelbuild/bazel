// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;

/**
 * A utility class to dynamically lookup TransitiveInfoProviders.
 */
public final class TransitiveInfoProviderCache {
  /**
   * A cache that stores the list of providers implemented by a given ConfiguredTarget class.
   */
  private static final
      LoadingCache<Class<?>, ImmutableList<Class<? extends TransitiveInfoProvider>>>
      PROVIDER_CLASSES_CACHE = CacheBuilder.newBuilder().concurrencyLevel(1).build(
          new CacheLoader<Class<?>, ImmutableList<Class<? extends TransitiveInfoProvider>>>() {
            @Override
            public ImmutableList<Class<? extends TransitiveInfoProvider>> load(Class<?> klass)
                throws Exception {
              List<Class<? extends TransitiveInfoProvider>> providers = new ArrayList<>();
              recursivelyCollectProviderClasses(klass, providers, Sets.<Class<?>>newHashSet());
              return ImmutableList.copyOf(providers);
            }

          });

  private TransitiveInfoProviderCache() {
  }

  /**
   * Returns a list of provider interfaces implemented by a given configured target.
   */
  @VisibleForTesting
  public static ImmutableList<Class<? extends TransitiveInfoProvider>> getProviderClasses(
      Class<?> klass) {
    try {
      return PROVIDER_CLASSES_CACHE.get(klass);
    } catch (ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Collects provider interfaces from base classes & interfaces (including itself) of a given
   * {@code ConfiguredTarget} class.
   *
   * <p>Provider interfaces are defined as interfaces which directly extend
   * {@link TransitiveInfoProvider}.
   *
   * @param klass the class to process
   * @param providers collection for storing provider interfaces that were found
   * @param seen classes that have already been examined
   */
  @SuppressWarnings("unchecked")  // Cast based on reflection
  private static void recursivelyCollectProviderClasses(Class<?> klass,
      List<Class<? extends TransitiveInfoProvider>> providers, Set<Class<?>> seen) {
    if (seen.contains(klass)) {
      return;
    }
    seen.add(klass);

    Class<?>[] interfaces = klass.getInterfaces();
    for (Class<?> iface : interfaces) {
      if (klass.isInterface() && iface.equals(TransitiveInfoProvider.class)) {
        // This unchecked cast is fine because we just checked its validity.
        providers.add((Class<? extends TransitiveInfoProvider>) klass);
        return;
      }
      recursivelyCollectProviderClasses(iface, providers, seen);
    }
    Class<?> superClass = klass.getSuperclass();
    if (superClass != null) {
      recursivelyCollectProviderClasses(superClass, providers, seen);
    }
  }
}
