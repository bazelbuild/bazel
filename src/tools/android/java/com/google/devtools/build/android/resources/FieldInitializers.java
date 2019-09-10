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
package com.google.devtools.build.android.resources;

import com.android.resources.ResourceType;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.common.collect.SortedSetMultimap;
import java.util.Collection;
import java.util.EnumMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Represents a collection of resource symbols and values suitable for writing java sources and
 * classes.
 */
public class FieldInitializers
    implements Iterable<Map.Entry<ResourceType, Collection<FieldInitializer>>> {

  private final Map<ResourceType, Collection<FieldInitializer>> initializers;

  private FieldInitializers(Map<ResourceType, Collection<FieldInitializer>> initializers) {
    this.initializers = initializers;
  }

  /** Creates a {@link FieldInitializers} copying the contents from a {@link Map}. */
  public static FieldInitializers copyOf(
      Map<ResourceType, Collection<FieldInitializer>> initializers) {
    Map<ResourceType, Collection<FieldInitializer>> deeplyImmutableInitializers =
        initializers.entrySet().stream()
            .collect(
                ImmutableListMultimap.flatteningToImmutableListMultimap(
                    Map.Entry::getKey, entry -> entry.getValue().stream()))
            .asMap();

    return new FieldInitializers(deeplyImmutableInitializers);
  }

  public static FieldInitializers mergedFrom(Collection<FieldInitializers> toMerge) {
    final Map<ResourceType, Collection<FieldInitializer>> merged =
        new EnumMap<>(ResourceType.class);

    for (ResourceType resourceType : ResourceType.values()) {
      ImmutableList<FieldInitializer> fieldInitializers =
          toMerge.stream()
              .flatMap(
                  fis -> fis.initializers.getOrDefault(resourceType, ImmutableList.of()).stream())
              .filter(distinctByKey(FieldInitializer::getFieldName))
              .collect(ImmutableList.toImmutableList());

      if (!fieldInitializers.isEmpty()) {
        merged.put(resourceType, fieldInitializers);
      }
    }

    return copyOf(merged);
  }

  private static <T> Predicate<T> distinctByKey(Function<? super T, ?> keyExtractor) {
    Set<Object> seen = ConcurrentHashMap.newKeySet();
    return t -> seen.add(keyExtractor.apply(t));
  }

  public FieldInitializers filter(FieldInitializers fieldsToWrite) {
    // Create a map to filter with.
    final SortedSetMultimap<ResourceType, String> symbolsToWrite =
        MultimapBuilder.enumKeys(ResourceType.class).treeSetValues().build();
    for (Map.Entry<ResourceType, Collection<FieldInitializer>> entry :
        fieldsToWrite.initializers.entrySet()) {
      for (FieldInitializer initializer : entry.getValue()) {
        symbolsToWrite.put(entry.getKey(), initializer.getFieldName());
      }
    }

    final Multimap<ResourceType, FieldInitializer> initializersToWrite =
        MultimapBuilder.enumKeys(ResourceType.class).arrayListValues().build();
    for (Map.Entry<ResourceType, Collection<String>> entry : symbolsToWrite.asMap().entrySet()) {
      // Resource type may be missing if resource overriding eliminates resources at the binary
      // level, which were originally present at the library level.
      for (FieldInitializer field : initializers.getOrDefault(entry.getKey(), ImmutableList.of())) {
        if (entry.getValue().contains(field.getFieldName())) {
          initializersToWrite.put(entry.getKey(), field);
        }
      }
    }
    return copyOf(initializersToWrite.asMap());
  }

  @Override
  public Iterator<Map.Entry<ResourceType, Collection<FieldInitializer>>> iterator() {
    return initializers.entrySet().iterator();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(FieldInitializers.class)
        .add("initializers", initializers)
        .toString();
  }
}
