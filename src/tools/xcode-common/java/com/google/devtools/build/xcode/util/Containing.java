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

package com.google.devtools.build.xcode.util;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.Multimap;

import java.util.Collection;
import java.util.Map;

/**
 * Provides methods that make checking for the presence of an item in a collection type-safe. For
 * instance, in {@code foo.containsKey(bar)}, where {@code foo} is a {@code Map<K, V>}, {@code bar}
 * can be a type other than {@code K} and may be {@code null}, in which case the method will just
 * return false (collections that allow null references may of course return true in the latter
 * case).
 * <p>
 * The methods in this class, such as {@link #key(Map, Object)}, will cause a compiler error if you
 * use the wrong type and throw a {@link NullPointerException} if you pass {@code null} for the
 * object whose presence to check. In the case where you want to check for {@code null} in a
 * collection, add a new method to this class, use the methods in the plain Collections API
 * (such as {@link Collection#contains(Object)}), or use the {@code Optional} type as the element of
 * the collection.
 * <p>
 * TODO(bazel-team): This class should either be simplified or eliminated when the
 * CollectionIncompatibleType feature is available:
 * https://code.google.com/p/error-prone/wiki/CollectionIncompatibleType
 */
public class Containing {
  private Containing() {
    throw new UnsupportedOperationException("static-only");
  }

  public static <K> boolean key(Map<K, ?> map, K key) {
    checkNotNull(key);
    return map.containsKey(key);
  }

  public static <K> boolean key(Multimap<K, ?> map, K key) {
    checkNotNull(key);
    return map.containsKey(key);
  }

  public static <E> boolean item(Collection<E> collection, E item) {
    checkNotNull(item);
    return collection.contains(item);
  }

  public static <V> boolean value(Map<?, V> map, V value) {
    checkNotNull(value);
    return map.containsValue(value);
  }

  public static <V> boolean value(Multimap<?, V> map, V value) {
    checkNotNull(value);
    return map.containsValue(value);
  }
}
