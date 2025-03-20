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
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Value;
import java.lang.reflect.Array;
import java.util.Map;
import java.util.Set;
import net.starlark.java.eval.Debug;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Structure;

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
    if (value instanceof String string) {
      return string;
    }
    return Starlark.repr(value);
  }

  private static boolean hasChildren(Object value) {
    if (value instanceof Map<?, ?> map) {
      return !map.isEmpty();
    }
    if (value instanceof Map.Entry) {
      return true;
    }
    if (value instanceof Iterable<?> iterable) {
      return iterable.iterator().hasNext();
    }
    if (value.getClass().isArray()) {
      return Array.getLength(value) > 0;
    }
    if (value instanceof Debug.ValueWithDebugAttributes) {
      return true;
    }
    if (value instanceof StarlarkInt) {
      return false;
    }
    if (value instanceof Structure || value instanceof StarlarkValue) {
      // assuming Structure's have at least one child as a temporary optimization
      // TODO(bazel-team): remove once child-listing logic is moved to StarlarkValue
      return true;
    }
    // fallback to assuming there are no children
    return false;
  }

  static ImmutableList<Value> getChildren(ThreadObjectMap objectMap, Object value) {
    if (value instanceof Map<?, ?> map) {
      return getChildren(objectMap, map.entrySet());
    }
    if (value instanceof Map.Entry<?, ?> mapEntry) {
      return getChildren(objectMap, mapEntry);
    }
    if (value instanceof Iterable<?> iterable) {
      return getChildren(objectMap, iterable);
    }
    if (value.getClass().isArray()) {
      return getArrayChildren(objectMap, value);
    }
    if (value instanceof Debug.ValueWithDebugAttributes) {
      return getDebugAttributes(objectMap, (Debug.ValueWithDebugAttributes) value);
    }
    // TODO(bazel-team): move child-listing logic to StarlarkValue where practical
    if (value instanceof Structure structure) {
      return getChildren(objectMap, structure);
    }
    if (value instanceof StarlarkValue starlarkValue) {
      return getChildren(objectMap, starlarkValue);
    }
    // fallback to assuming there are no children
    return ImmutableList.of();
  }

  private static ImmutableList<Value> getChildren(
      ThreadObjectMap objectMap, Structure classObject) {
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
      ThreadObjectMap objectMap, StarlarkValue starlarkValue) {
    StarlarkSemantics semantics = StarlarkSemantics.DEFAULT; // TODO(adonovan): obtain from thread.
    // TODO(adonovan): would the debugger be content with Starlark.{dir,getattr}
    // instead of getAnnotatedField{,Names}, if we filtered out BuiltinCallables?
    Set<String> fieldNames;
    try {
      fieldNames = Starlark.getAnnotatedFieldNames(semantics, starlarkValue);
    } catch (IllegalArgumentException e) {
      // silently return no children
      return ImmutableList.of();
    }
    ImmutableList.Builder<Value> children = ImmutableList.builder();
    for (String fieldName : fieldNames) {
      try {
        children.add(
            getValueProto(
                objectMap,
                fieldName,
                Starlark.getAnnotatedField(semantics, starlarkValue, fieldName)));
      } catch (EvalException | InterruptedException | IllegalArgumentException e) {
        // silently ignore errors
      }
    }
    return children.build();
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

  private static ImmutableList<Value> getDebugAttributes(
      ThreadObjectMap objectMap, Debug.ValueWithDebugAttributes value) {
    ImmutableList.Builder<Value> attributes = ImmutableList.builder();
    for (Debug.DebugAttribute attr : value.getDebugAttributes()) {
      attributes.add(getValueProto(objectMap, attr.name, attr.value));
    }
    return attributes.build();
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
