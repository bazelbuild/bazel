// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildtool;

import static com.google.devtools.build.lib.util.ShallowObjectSizeComputer.getArraySize;
import static com.google.devtools.build.lib.util.ShallowObjectSizeComputer.getShallowSize;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.collect.CompactImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.MemoryAccountant;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.DomainSpecificTraverser;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.Traversal;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** An object traverser that handles common collection classes. */
public class CollectionObjectTraverser
    implements DomainSpecificTraverser, MemoryAccountant.Measurer {
  private static final String NESTEDSET_ARRAY = "Object[] NestedSet";
  private static final Field NESTEDSET_CHILDREN;

  static {
    try {
      NESTEDSET_CHILDREN = NestedSet.class.getDeclaredField("children");
      NESTEDSET_CHILDREN.setAccessible(true);
    } catch (NoSuchFieldException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public boolean isInterned(Object o) {
    return false;
  }

  @Override
  public long maybeGetShallowSize(Object o) {
    return switch (o) {
      case List<?> l -> getShallowSize(l) + getArraySize(l.size(), Object.class);
      case Set<?> s -> getShallowSize(s) + getArraySize(s.size(), Object.class);
      case Map<?, ?> m ->
          // 32 is an estimate for the per-entry overhead of Map and Multimap
          getShallowSize(m) + m.size() * 32L;
      case Multimap<?, ?> mm -> getShallowSize(mm) + mm.size() * 32L;
      case CompactImmutableMap<?, ?> cim ->
          // For CompactImmutableMap, we ignore OffsetTable: it's interned so it's difficult to
          // assign to any one SkyValue and there aren't supposed to be many of those anyway.
          getShallowSize(cim) + getArraySize(cim.size(), Object.class);

      default -> -1;
    };
  }

  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public boolean maybeTraverse(Object o, Traversal traversal) {
    switch (o) {
      case List<?> l -> {
        traversal.objectFound(l, "List");
        for (Object m : l) {
          traversal.edgeFound(m, null);
        }

        return true;
      }

      case Set<?> s -> {
        traversal.objectFound(s, "Set");
        for (Object m : s) {
          traversal.edgeFound(m, null);
        }

        return true;
      }

      case Map<?, ?> m -> {
        traversal.objectFound(m, "Map");
        for (Map.Entry<?, ?> e : m.entrySet()) {
          traversal.edgeFound(e.getKey(), null);
          traversal.edgeFound(e.getValue(), null);
        }

        return true;
      }

      case Multimap<?, ?> mm -> {
        traversal.objectFound(mm, "Multimap");
        for (Map.Entry<?, ?> e : mm.entries()) {
          traversal.edgeFound(e.getKey(), null);
          traversal.edgeFound(e.getValue(), null);
        }

        return true;
      }

      case CompactImmutableMap cim -> {
        traversal.objectFound(cim, "CompactImmutableMap");
        for (Object k : cim) {
          traversal.edgeFound(k, null);
          traversal.edgeFound(cim.get(k), null);
        }
        return true;
      }

      case NestedSet<?> ns -> {
        traversal.objectFound(ns, "NestedSet");
        Object children;
        try {
          children = NESTEDSET_CHILDREN.get(ns);
        } catch (IllegalArgumentException | IllegalAccessException e) {
          throw new IllegalStateException(e);
        }

        if (children instanceof Object[]) {
          traversal.edgeFound(children, NESTEDSET_ARRAY);
        } else {
          traversal.edgeFound(children, null);
        }

        return true;
      }

      default -> {
        return false;
      }
    }
  }

  @Override
  public boolean admit(Object o) {
    return true;
  }

  @Nullable
  @Override
  public String contextForArrayItem(Object from, String fromContext, Object to) {
    return null;
  }

  @Nullable
  @Override
  public String contextForField(Object from, String fromContext, Field field, Object to) {
    return null;
  }

  @Nullable
  @Override
  public ImmutableSet<String> ignoredFields(Class<?> clazz) {
    return null;
  }
}
