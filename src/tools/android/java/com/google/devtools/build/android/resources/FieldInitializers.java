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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.MultimapBuilder;
import com.google.common.collect.SortedSetMultimap;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/** A wrapper for a {@link Collection} of {@link FieldInitializer}s. */
public class FieldInitializers
    implements Iterable<Entry<ResourceType, Collection<FieldInitializer>>> {

  private final Map<ResourceType, Collection<FieldInitializer>> initializers;

  private FieldInitializers(Map<ResourceType, Collection<FieldInitializer>> initializers) {
    this.initializers = initializers;
  }

  /** Creates a {@link FieldInitializers} copying the contents from a {@link Map}. */
  public static FieldInitializers copyOf(
      Map<ResourceType, Collection<FieldInitializer>> initializers) {
    return new FieldInitializers(ImmutableMap.copyOf(initializers));
  }

  public Iterable<Entry<ResourceType, Collection<FieldInitializer>>> filter(
      Map<ResourceType, Set<String>> symbolsToWrite) {
    final SortedSetMultimap<ResourceType, FieldInitializer> initializersToWrite =
        MultimapBuilder.enumKeys(ResourceType.class).treeSetValues().build();
    for (Entry<ResourceType, Set<String>> entry : symbolsToWrite.entrySet()) {
      // Resource type may be missing if resource overriding eliminates resources at the binary
      // level, which were originally present at the library level.
      if (initializers.containsKey(entry.getKey())) {
        for (FieldInitializer field : initializers.get(entry.getKey())) {
          if (field.nameIsIn(entry.getValue())) {
            initializersToWrite.put(entry.getKey(), field);
          }
        }
      }
    }
    return initializersToWrite.asMap().entrySet();
  }

  @Override
  public Iterator<Entry<ResourceType, Collection<FieldInitializer>>> iterator() {
    return initializers.entrySet().iterator();
  }
}
