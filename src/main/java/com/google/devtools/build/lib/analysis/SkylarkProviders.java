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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A helper class for transitive infos provided by Skylark rule implementations.
 */
@Immutable
public final class SkylarkProviders implements TransitiveInfoProvider {
  private final ImmutableMap<String, Object> skylarkProviders;

  SkylarkProviders(ImmutableMap<String, Object> skylarkProviders) {
    Preconditions.checkNotNull(skylarkProviders);
    this.skylarkProviders = skylarkProviders;
  }

  /**
   * Returns the keys for the Skylark providers.
   */
  public ImmutableCollection<String> getKeys() {
    return skylarkProviders.keySet();
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

  /**
   * Merges skylark providers. The set of providers must be disjoint.
   *
   * @param providers providers to merge {@code this} with.
   */

  public static SkylarkProviders merge(List<SkylarkProviders> providers) {
    if (providers.size() == 0) {
      return null;
    }
    if (providers.size() == 1) {
      return providers.get(0);
    }

    ImmutableMap.Builder<String, Object> resultBuilder = new ImmutableMap.Builder<>();
    Set<String> seenKeys = new HashSet<>();
    for (SkylarkProviders provider : providers) {
      for (String key : provider.skylarkProviders.keySet()) {
        if (!seenKeys.add(key)) {
          // TODO(dslomov): add better diagnostics.
          throw new IllegalStateException("Skylark provider " + key + " provided twice");
        }

        resultBuilder.put(key, provider.getValue(key));
      }
    }
    return new SkylarkProviders(resultBuilder.build());
  }
}
