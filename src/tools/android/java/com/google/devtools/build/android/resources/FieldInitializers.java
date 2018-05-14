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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Sets;
import java.util.Collection;
import java.util.EnumMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;

/**
 * Represents a collection of resource symbols and values suitable for writing java sources and
 * classes.
 */
public class FieldInitializers
    implements Iterable<Map.Entry<ResourceType, Map<String, FieldInitializer>>> {

  private final Map<ResourceType, Map<String, FieldInitializer>> initializers;

  private FieldInitializers(Map<ResourceType, Map<String, FieldInitializer>> initializers) {
    this.initializers = initializers;
  }

  /** Creates a {@link FieldInitializers} copying the contents from a {@link Map}. */
  public static FieldInitializers copyOf(
      Map<ResourceType, Map<String, FieldInitializer>> initializers) {
    return new FieldInitializers(ImmutableMap.copyOf(initializers));
  }

  public static FieldInitializers mergedFrom(Collection<FieldInitializers> toMerge) {
    final Map<ResourceType, Map<String, FieldInitializer>> merged =
        new EnumMap<>(ResourceType.class);
    for (FieldInitializers mergee : toMerge) {
      for (Map.Entry<ResourceType, Map<String, FieldInitializer>> entry : mergee) {
        final Map<String, FieldInitializer> fieldMap =
            merged.containsKey(entry.getKey())
                ? merged.get(entry.getKey())
                : new TreeMap<String, FieldInitializer>();
        merged.put(entry.getKey(), fieldMap);
        for (Map.Entry<String, FieldInitializer> field : entry.getValue().entrySet()) {
          fieldMap.put(field.getKey(), field.getValue());
        }
      }
    }
    return copyOf(merged);
  }

  public Iterable<Map.Entry<ResourceType, Map<String, FieldInitializer>>> filter(
      FieldInitializers fieldsToWrite) {
    Map<ResourceType, Map<String, FieldInitializer>> initializersToWrite =
        new EnumMap<>(ResourceType.class);

    for (Map.Entry<ResourceType, Map<String, FieldInitializer>> entry :
        fieldsToWrite.initializers.entrySet()) {
      if (initializers.containsKey(entry.getKey())) {
        final Map<String, FieldInitializer> valueFields = initializers.get(entry.getKey());
        final ImmutableMap.Builder<String, FieldInitializer> fields =
            ImmutableSortedMap.naturalOrder();
        // Resource type may be missing if resource overriding eliminates resources at the binary
        // level, which were originally present at the library level.
        for (String fieldName :
            Sets.intersection(entry.getValue().keySet(), valueFields.keySet())) {
          fields.put(fieldName, valueFields.get(fieldName));
        }
        initializersToWrite.put(entry.getKey(), fields.build());
      }
    }

    return initializersToWrite.entrySet();
  }

  @Override
  public Iterator<Map.Entry<ResourceType, Map<String, FieldInitializer>>> iterator() {
    return initializers.entrySet().iterator();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(FieldInitializers.class)
        .add("initializers", initializers)
        .toString();
  }
}
