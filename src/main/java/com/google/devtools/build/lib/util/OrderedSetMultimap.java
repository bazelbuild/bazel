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

package com.google.devtools.build.lib.util;

import com.google.common.collect.ForwardingSetMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;

/**
 * A {@code Multimap} that cannot hold duplicate key-value pairs, returns {@code keySet},
 * {@code keys}, and {@code asMap} collections that iterate through the keys in the order they were
 * first added, and maintains the insertion ordering of values for a given key. See the
 * {@link Multimap} documentation for information common to all multimaps.
 */
public final class OrderedSetMultimap<K, V> extends ForwardingSetMultimap<K, V> {
  private final LinkedHashMultimap<K, V> delegate = LinkedHashMultimap.<K, V>create();

  @Override
  protected SetMultimap delegate() {
    return delegate;
  }

  public static <K, V> OrderedSetMultimap<K, V> create() {
    return new OrderedSetMultimap<K, V>();
  }
}
