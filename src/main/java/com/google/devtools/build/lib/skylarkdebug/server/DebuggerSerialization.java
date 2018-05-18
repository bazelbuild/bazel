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

package com.google.devtools.build.lib.skylarkdebug.server;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.lang.reflect.Array;
import java.util.Map;

/** Helper class for creating {@link SkylarkDebuggingProtos.Value} from skylark objects. */
final class DebuggerSerialization {

  static Value getValueProto(String label, Object value) {
    // TODO(bazel-team): prune cycles, and provide a way to limit breadth/depth of children reported
    return Value.newBuilder()
        .setLabel(label)
        // TODO(bazel-team): omit type details for non-Skylark values
        .setType(EvalUtils.getDataTypeName(value))
        .setDescription(Printer.repr(value))
        .addAllChild(getChildren(value))
        .build();
  }

  private static Value errorValue(String errorMessage) {
    return Value.newBuilder().setLabel("Error").setDescription(errorMessage).build();
  }

  private static ImmutableList<Value> getChildren(Object value) {
    // TODO(bazel-team): move child-listing logic to SkylarkValue where practical
    if (value instanceof ClassObject) {
      return getChildren((ClassObject) value);
    }
    if (value instanceof SkylarkNestedSet) {
      return getChildren((SkylarkNestedSet) value);
    }
    if (value instanceof NestedSetView) {
      return getChildren((NestedSetView) value);
    }
    if (value instanceof Map) {
      return getChildren(((Map) value).entrySet());
    }
    if (value instanceof Map.Entry) {
      return getChildren((Map.Entry) value);
    }
    if (value instanceof Iterable) {
      return getChildren((Iterable) value);
    }
    if (value.getClass().isArray()) {
      return getArrayChildren(value);
    }
    // fallback to assuming there are no children
    return ImmutableList.of();
  }

  private static ImmutableList<Value> getChildren(ClassObject classObject) {
    ImmutableList.Builder<Value> builder = ImmutableList.builder();
    ImmutableList<String> keys;
    try {
      keys = Ordering.natural().immutableSortedCopy(classObject.getFieldNames());
    } catch (EvalException e) {
      return ImmutableList.of(errorValue("Error retrieving field names: " + e.getMessage()));
    }
    for (String key : keys) {
      Object value;
      try {
        value = classObject.getValue(key);
      } catch (EvalException e) {
        return ImmutableList.of(
            errorValue(
                String.format("Error retrieving value for field '%s': %s", key, e.getMessage())));
      }
      if (value != null) {
        builder.add(getValueProto(key, value));
      }
    }
    return builder.build();
  }

  private static ImmutableList<Value> getChildren(SkylarkNestedSet nestedSet) {
    Class<?> type = nestedSet.getContentType().getType();
    return ImmutableList.<Value>builder()
        .add(
            Value.newBuilder()
                .setLabel("order")
                .setType("Traversal order")
                .setDescription(nestedSet.getOrder().getSkylarkName())
                .build())
        .addAll(getChildren(new NestedSetView<>(nestedSet.getSet(type))))
        .build();
  }

  private static ImmutableList<Value> getChildren(NestedSetView<?> nestedSet) {
    return ImmutableList.of(
        getValueProto("directs", nestedSet.directs()),
        getValueProto("transitives", nestedSet.transitives()));
  }

  private static ImmutableList<Value> getChildren(Map.Entry<?, ?> entry) {
    return ImmutableList.of(
        getValueProto("key", entry.getKey()), getValueProto("value", entry.getValue()));
  }

  private static ImmutableList<Value> getChildren(Iterable<?> iterable) {
    ImmutableList.Builder<Value> builder = ImmutableList.builder();
    int index = 0;
    for (Object value : iterable) {
      builder.add(getValueProto(String.format("[%d]", index++), value));
    }
    return builder.build();
  }

  private static ImmutableList<Value> getArrayChildren(Object array) {
    ImmutableList.Builder<Value> builder = ImmutableList.builder();
    int index = 0;
    for (int i = 0; i < Array.getLength(array); i++) {
      builder.add(getValueProto(String.format("[%d]", index++), Array.get(array, i)));
    }
    return builder.build();
  }
}
