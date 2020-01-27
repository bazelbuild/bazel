// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.Interner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * Strong {@link Interner} that also exposes whether there is a canonical representative for the
 * given sample object via {@link #getCanonical}.
 */
public class InternerWithPresenceCheck<T> implements Interner<T> {
  private final ConcurrentMap<T, T> map = new ConcurrentHashMap<>();

  @Override
  public T intern(T sample) {
    T canonical = map.putIfAbsent(checkNotNull(sample), sample);
    return (canonical == null) ? sample : canonical;
  }

  /**
   * Returns the canonical representative for {@code sample} if it is present. Unlike {@link
   * #intern}, does not store {@code sample}. In other words, this method does not mutate the
   * interner.
   */
  @Nullable
  T getCanonical(T sample) {
    return map.get(sample);
  }
}
