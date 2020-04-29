// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdebug.server;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.CallUtils;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.lang.reflect.Array;
import java.util.Map;
import java.util.Set;

/** Helper class for creating {@link StarlarkDebuggingProtos.Value} from Starlark objects. */
final class DebuggerSerialization {

  static Value getValueProto(ThreadObjectMap objectMap, String label, Object value) {
    // TODO(bazel-team): prune cycles, and provide a way to limit breadth/depth of children reported
    boolean hasChildren = hasChildren(value);
    return Value.newBuilder()
        .setLabel(label)
        // TODO(bazel-team): omit type details for non-Starlark values
        .setType(Starlark.type(value))
        .setDescription(getDescription(value))
        .setHasChildren(hasChildren)
        .setId(hasChildren ? objectMap.registerValue(value) : 0)
        .build();
  }

  private static String getDescription(Object value) {
    if (value instanceof String) {
      return (String) value;
    }
    return Starlark.repr(value);
  }

  private static boolean hasChildren(Object value) {
    if (value instanceof Depset) {
      return true;
    }
    if (value instanceof NestedSetView) {
      return true;
    }
    if (value instanceof Map) {
      return !((Map) value).isEmpty();
    }
    if (value instanceof Map.Entry) {
      return true;
    }
    if (value instanceof Iterable) {
      return ((Iterable) value).iterator().hasNext();
    }
    if (value.getClass().isArray()) {
      return Array.getLength(value) > 0;
    }
    if (value instanceof ClassObject || value instanceof StarlarkValue) {
      // assuming ClassObject's have at least one child as a temporary optimization
      // TODO(bazel-team): remove once child-listing logic is moved to StarlarkValue
      return true;
    }
    // fallback to assuming there are no children
    return false;
  }

  static ImmutableList<Value> getChildren(ThreadObjectMap objectMap, Object value) {
    if (value instanceof Depset) {
      return getChildren(objectMap, (Depset) value);
    }
    if (value instanceof NestedSetView) {
      return getChildren(objectMap, (NestedSetView) value);
    }
    if (value instanceof Map) {
      return getChildren(objectMap, ((Map) value).entrySet());
    }
    if (value instanceof Map.Entry) {
      return getChildren(objectMap, (Map.Entry) value);
    }
    if (value instanceof Iterable) {
      return getChildren(objectMap, (Iterable) value);
    }
    if (value.getClass().isArray()) {
      return getArrayChildren(objectMap, value);
    }
    // TODO(bazel-team): move child-listing logic to StarlarkValue where practical
    if (value instanceof ClassObject) {
      return getChildren(objectMap, (ClassObject) value);
    }
    if (value instanceof StarlarkValue) {
      return getChildren(objectMap, (StarlarkValue) value);
    }
    // fallback to assuming there are no children
    return ImmutableList.of();
  }

  private static ImmutableList<Value> getChildren(
      ThreadObjectMap objectMap, ClassObject classObject) {
    ImmutableList.Builder<Value> builder = ImmutableList.builder();
    for (String key : Ordering.natural().immutableSortedCopy(classObject.getFieldNames())) {
      try {
        Object value = classObject.getValue(key);
        if (value != null) {
          builder.add(getValueProto(objectMap, key, value));
        }
      } catch (EvalException | IllegalArgumentException e) {
        // silently ignore errors
      }
    }
    return builder.build();
  }

  private static ImmutableList<Value> getChildren(
      ThreadObjectMap objectMap, StarlarkValue skylarkValue) {
    StarlarkSemantics semantics =
        StarlarkSemantics.DEFAULT_SEMANTICS; // TODO(adonovan): obtain from thread.
    Set<String> fieldNames;
    try {
      fieldNames = CallUtils.getFieldNames(semantics, skylarkValue);
    } catch (IllegalArgumentException e) {
      // silently return no children
      return ImmutableList.of();
    }
    ImmutableList.Builder<Value> children = ImmutableList.builder();
    for (String fieldName : fieldNames) {
      try {
        children.add(
            getValueProto(
                objectMap, fieldName, CallUtils.getField(semantics, skylarkValue, fieldName)));
      } catch (EvalException | InterruptedException | IllegalArgumentException e) {
        // silently ignore errors
      }
    }
    return children.build();
  }

  private static ImmutableList<Value> getChildren(ThreadObjectMap objectMap, Depset nestedSet) {
    return ImmutableList.<Value>builder()
        .add(
            Value.newBuilder()
                .setLabel("order")
                .setType("Traversal order")
                .setDescription(nestedSet.getOrder().getSkylarkName())
                .build())
        .addAll(getChildren(objectMap, new NestedSetView<>(nestedSet.getSet())))
        .build();
  }

  private static ImmutableList<Value> getChildren(
      ThreadObjectMap objectMap, NestedSetView<?> nestedSet) {
    return ImmutableList.of(
        getValueProto(objectMap, "directs", nestedSet.directs()),
        getValueProto(objectMap, "transitives", nestedSet.transitives()));
  }

  private static ImmutableList<Value> getChildren(
      ThreadObjectMap objectMap, Map.Entry<?, ?> entry) {
    return ImmutableList.of(
        getValueProto(objectMap, "key", entry.getKey()),
        getValueProto(objectMap, "value", entry.getValue()));
  }

  private static ImmutableList<Value> getChildren(ThreadObjectMap objectMap, Iterable<?> iterable) {
    ImmutableList.Builder<Value> builder = ImmutableList.builder();
    int index = 0;
    for (Object value : iterable) {
      builder.add(getValueProto(objectMap, String.format("[%d]", index++), value));
    }
    return builder.build();
  }

  private static ImmutableList<Value> getArrayChildren(ThreadObjectMap objectMap, Object array) {
    ImmutableList.Builder<Value> builder = ImmutableList.builder();
    int index = 0;
    for (int i = 0; i < Array.getLength(array); i++) {
      builder.add(getValueProto(objectMap, String.format("[%d]", index++), Array.get(array, i)));
    }
    return builder.build();
  }
}
