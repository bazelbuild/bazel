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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.MergedConfiguredTarget.DuplicateException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A helper class for transitive infos provided by Skylark rule implementations.
 */
@Immutable
public final class SkylarkProviders implements TransitiveInfoProvider {
  private final ImmutableMap<ClassObjectConstructor.Key, SkylarkClassObject>
      declaredProviders;
  private final ImmutableMap<String, Object> skylarkProviders;

  SkylarkProviders(
      ImmutableMap<String, Object> skylarkProviders,
      ImmutableMap<ClassObjectConstructor.Key, SkylarkClassObject> declaredProviders) {
    this.declaredProviders = Preconditions.checkNotNull(declaredProviders);
    this.skylarkProviders = Preconditions.checkNotNull(skylarkProviders);
  }

  public void init(ConfiguredTarget target) {
    for (Object o : skylarkProviders.values()) {
      if (o instanceof SkylarkApiProvider) {
        ((SkylarkApiProvider) o).init(target);
      }
    }
  }

  /**
   * Returns the keys for the Skylark providers.
   */
  public ImmutableCollection<String> getKeys() {
    return skylarkProviders.keySet();
  }

  /** Returns the keys for the declared providers. */
  public ImmutableCollection<ClassObjectConstructor.Key> getDeclaredProviderKeys() {
    return declaredProviders.keySet();
  }

  /**
   * Returns a Skylark provider; "key" must be one from {@link #getKeys()}.
   */
  public Object getValue(String key) {
    return skylarkProviders.get(key);
  }

  /**
   * Returns a Skylark provider and try to cast it into the specified type
   */
  public <TYPE> TYPE getValue(String key, Class<TYPE> type) throws EvalException {
    Object obj = skylarkProviders.get(key);
    if (obj == null) {
      return null;
    }
    SkylarkType.checkType(obj, type, key);
    return type.cast(obj);
  }

  public SkylarkClassObject getDeclaredProvider(ClassObjectConstructor.Key key) {
    return declaredProviders.get(key);
  }

  public Object get(SkylarkProviderIdentifier id) {
    if (id.isLegacy()) {
      return getValue(id.getLegacyId());
    }
    return getDeclaredProvider(id.getKey());
  }


  private static final Function<SkylarkProviders, Map<String, Object>>
      SKYLARK_PROVIDERS_MAP_FUNCTION = new Function<SkylarkProviders, Map<String, Object>>() {
        @Override
        public Map<String, Object> apply(SkylarkProviders skylarkProviders) {
          return skylarkProviders.skylarkProviders;
        }
      };

  public static final Function<SkylarkProviders,
                               Map<ClassObjectConstructor.Key, SkylarkClassObject>>
      DECLARED_PROVIDERS_MAP_FUNCTION =
      new Function<SkylarkProviders, Map<ClassObjectConstructor.Key, SkylarkClassObject>>() {
        @Override
        public Map<ClassObjectConstructor.Key, SkylarkClassObject> apply(
            SkylarkProviders skylarkProviders) {
          return skylarkProviders.declaredProviders;
        }
      };

  /**
   * Merges skylark providers. The set of providers must be disjoint.
   *
   * @param premergedProviders providers that has already been merged. They will
   *        be put into the result as-is and their presence will be ignored among {@code providers}.
   * @param providers providers to merge {@code this} with.
   */
  public static SkylarkProviders merge(
      ImmutableMap<String, Object> premergedLegacyProviders,
      ImmutableMap<SkylarkClassObjectConstructor.Key, SkylarkClassObject> premergedProviders,
      List<SkylarkProviders> providers)
      throws DuplicateException {
    if (premergedProviders.size() == 0 && providers.size() == 0) {
      return null;
    }
    if (premergedProviders.size() == 0 && providers.size() == 1) {
      return providers.get(0);
    }

    ImmutableMap<String, Object> skylarkProviders = mergeMaps(providers,
        SKYLARK_PROVIDERS_MAP_FUNCTION,
        premergedLegacyProviders);

    ImmutableMap<ClassObjectConstructor.Key, SkylarkClassObject> declaredProviders =
        mergeMaps(providers, DECLARED_PROVIDERS_MAP_FUNCTION,
            premergedProviders);

    return new SkylarkProviders(skylarkProviders, declaredProviders);
  }

  private static <K, V> ImmutableMap<K, V> mergeMaps(List<SkylarkProviders> providers,
      Function<SkylarkProviders, Map<K, V>> mapGetter, Map<K, V> premerged)
      throws DuplicateException {
    Set<K> seenKeys = new HashSet<>();
    ImmutableMap.Builder<K, V> resultBuilder = ImmutableMap.builder();
    resultBuilder.putAll(premerged);
    for (SkylarkProviders provider : providers) {
      Map<K, V> map = mapGetter.apply(provider);
      for (K key : map.keySet()) {
        if (premerged.containsKey(key)) {
          continue;
        }
        if (!seenKeys.add(key)) {
          // TODO(dslomov): add better diagnostics.
          throw new DuplicateException("Provider " + key + " provided twice");
        }

        V v = map.get(key);
        if (v != null) {
          resultBuilder.put(key, v);
        }
      }
    }
    return resultBuilder.build();
  }
}
