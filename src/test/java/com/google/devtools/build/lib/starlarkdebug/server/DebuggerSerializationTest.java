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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link DebuggerSerialization}. */
@RunWith(JUnit4.class)
public final class DebuggerSerializationTest {

  private final ThreadObjectMap dummyObjectMap = new ThreadObjectMap();

  /**
   * Returns the {@link Value} proto message corresponding to the given object and label. Subsequent
   * calls may return values with different IDs.
   */
  private Value getValueProto(String label, Object value) {
    return DebuggerSerialization.getValueProto(dummyObjectMap, label, value);
  }

  private ImmutableList<Value> getChildren(Value value) {
    Object object = dummyObjectMap.getValue(value.getId());
    return object != null
        ? DebuggerSerialization.getChildren(dummyObjectMap, object)
        : ImmutableList.of();
  }

  @Test
  public void testSimpleNestedSet() {
    Set<String> children = ImmutableSet.of("a", "b");
    Depset set =
        Depset.of(
            Depset.ElementType.STRING, NestedSetBuilder.stableOrder().addAll(children).build());

    Value value = getValueProto("name", set);

    assertTypeAndDescription(set, value);
    assertThat(value.getHasChildren()).isTrue();
    assertThat(value.getLabel()).isEqualTo("name");

    List<Value> childValues = getChildren(value);

    assertThat(childValues.get(0))
        .isEqualTo(
            Value.newBuilder()
                .setLabel("order")
                .setType("Traversal order")
                .setDescription("default")
                .build());
    assertEqualIgnoringTypeDescriptionAndId(childValues.get(1), getValueProto("directs", children));
    assertEqualIgnoringTypeDescriptionAndId(
        childValues.get(2), getValueProto("transitives", ImmutableList.of()));
  }

  @Test
  public void testNestedSetWithNestedChildren() {
    NestedSet<String> innerNestedSet =
        NestedSetBuilder.<String>stableOrder().add("inner1").add("inner2").build();
    ImmutableSet<String> directChildren = ImmutableSet.of("a", "b");
    Depset outerSet =
        Depset.of(
            Depset.ElementType.STRING,
            NestedSetBuilder.<String>linkOrder()
                .addAll(directChildren)
                .addTransitive(innerNestedSet)
                .build());

    Value value = getValueProto("name", outerSet);
    List<Value> childValues = getChildren(value);

    assertTypeAndDescription(outerSet, value);
    assertThat(childValues).hasSize(3);
    assertThat(childValues.get(0))
        .isEqualTo(
            Value.newBuilder()
                .setLabel("order")
                .setType("Traversal order")
                .setDescription("topological")
                .build());
    assertEqualIgnoringTypeDescriptionAndId(
        childValues.get(1), getValueProto("directs", directChildren));
    assertEqualIgnoringTypeDescriptionAndId(
        childValues.get(2),
        getValueProto("transitives", ImmutableList.of(new NestedSetView<>(innerNestedSet))));
  }

  @Test
  public void testSimpleMap() {
    Map<String, Integer> map = ImmutableMap.of("a", 1, "b", 2);

    Value value = getValueProto("name", map);
    List<Value> childValues = getChildren(value);

    assertTypeAndDescription(map, value);
    assertThat(childValues).hasSize(2);
    assertThat(childValues.get(0).getLabel()).isEqualTo("[0]");
    assertThat(getChildren(childValues.get(0)))
        .isEqualTo(ImmutableList.of(getValueProto("key", "a"), getValueProto("value", 1)));
    assertThat(childValues.get(1).getLabel()).isEqualTo("[1]");
    assertThat(getChildren(childValues.get(1)))
        .isEqualTo(ImmutableList.of(getValueProto("key", "b"), getValueProto("value", 2)));
  }

  @Test
  public void testNestedMap() {
    Set<String> set = ImmutableSet.of("a", "b");
    Map<String, Object> map = ImmutableMap.of("a", set);

    Value value = getValueProto("name", map);
    List<Value> childValues = getChildren(value);

    assertTypeAndDescription(map, value);
    assertThat(childValues).hasSize(1);
    assertThat(childValues.get(0).getLabel()).isEqualTo("[0]");
    assertThat(clearIds(getChildren(childValues.get(0))))
        .isEqualTo(
            ImmutableList.of(getValueProto("key", "a"), clearId(getValueProto("value", set))));
  }

  @Test
  public void testSimpleIterable() {
    Iterable<Integer> iter = ImmutableList.of(1, 2);

    Value value = getValueProto("name", iter);
    List<Value> childValues = getChildren(value);

    assertTypeAndDescription(iter, value);
    assertThat(childValues).hasSize(2);
    assertThat(childValues.get(0)).isEqualTo(getValueProto("[0]", 1));
    assertThat(childValues.get(1)).isEqualTo(getValueProto("[1]", 2));
  }

  @Test
  public void testNestedIterable() {
    Iterable<Object> iter = ImmutableList.of(ImmutableList.of(1, 2));

    Value value = getValueProto("name", iter);
    List<Value> childValues = getChildren(value);

    assertTypeAndDescription(iter, value);
    assertThat(childValues).hasSize(1);
    assertValuesEqualIgnoringId(childValues.get(0), getValueProto("[0]", ImmutableList.of(1, 2)));
  }

  @Test
  public void testSimpleArray() {
    int[] array = new int[] {1, 2};

    Value value = getValueProto("name", array);
    List<Value> childValues = getChildren(value);

    assertTypeAndDescription(array, value);
    assertThat(childValues).hasSize(2);
    assertThat(childValues.get(0)).isEqualTo(getValueProto("[0]", 1));
    assertThat(childValues.get(1)).isEqualTo(getValueProto("[1]", 2));
  }

  @Test
  public void testNestedArray() {
    Object[] array = new Object[] {1, ImmutableList.of(2, 3)};

    Value value = getValueProto("name", array);
    List<Value> childValues = getChildren(value);

    assertTypeAndDescription(array, value);
    assertThat(childValues).hasSize(2);
    assertThat(childValues.get(0)).isEqualTo(getValueProto("[0]", 1));
    assertValuesEqualIgnoringId(childValues.get(1), getValueProto("[1]", ImmutableList.of(2, 3)));
  }

  @Test
  public void testUnrecognizedObjectOrSkylarkPrimitiveHasNoChildren() {
    assertThat(getValueProto("name", 1).getHasChildren()).isFalse();
    assertThat(getValueProto("name", "string").getHasChildren()).isFalse();
    assertThat(getValueProto("name", new Object()).getHasChildren()).isFalse();
  }

  @Test
  public void testStarlarkValue() {
    DummyType dummy = new DummyType();

    Value value = getValueProto("name", dummy);
    assertTypeAndDescription(dummy, value);
    assertThat(getChildren(value)).containsExactly(getValueProto("bool", true));
  }

  private static class DummyType implements StarlarkValue {
    @Override
    public void repr(Printer printer) {
      printer.append("DummyType");
    }

    @SkylarkCallable(name = "bool", doc = "Returns True", structField = true)
    public boolean bool() {
      return true;
    }

    public boolean anotherMethod() {
      return false;
    }
  }

  @Test
  public void testSkipSkylarkCallableThrowingException() {
    DummyTypeWithException dummy = new DummyTypeWithException();

    Value value = getValueProto("name", dummy);
    assertTypeAndDescription(dummy, value);
    assertThat(getChildren(value)).containsExactly(getValueProto("bool", true));
  }

  private static class DummyTypeWithException implements StarlarkValue {
    @Override
    public void repr(Printer printer) {
      printer.append("DummyTypeWithException");
    }

    @SkylarkCallable(name = "bool", doc = "Returns True", structField = true)
    public boolean bool() {
      return true;
    }

    @SkylarkCallable(name = "invalid", doc = "Throws exception!", structField = true)
    public boolean invalid() {
      throw new IllegalArgumentException();
    }

    public boolean anotherMethod() {
      return false;
    }
  }

  private static void assertTypeAndDescription(Object object, Value value) {
    assertThat(value.getType()).isEqualTo(Starlark.type(object));
    assertThat(value.getDescription()).isEqualTo(Starlark.repr(object));
  }

  /**
   * Type, description, and ID are implementation dependent (e.g. NestedSetView#directs returns a
   * list instead of a set if there are no duplicates, which changes both 'type' and 'description').
   */
  private void assertEqualIgnoringTypeDescriptionAndId(Value value1, Value value2) {
    assertThat(value1.getLabel()).isEqualTo(value2.getLabel());

    List<Value> children1 = getChildren(value1);
    List<Value> children2 = getChildren(value2);

    assertThat(children1).hasSize(children2.size());
    for (int i = 0; i < children1.size(); i++) {
      assertEqualIgnoringTypeDescriptionAndId(children1.get(i), children2.get(i));
    }
  }

  private void assertValuesEqualIgnoringId(Value value1, Value value2) {
    assertThat(clearId(value1)).isEqualTo(clearId(value2));
  }

  private Value clearId(Value value) {
    return value.toBuilder().clearId().build();
  }

  private List<Value> clearIds(List<Value> values) {
    return values.stream().map(this::clearId).collect(Collectors.toList());
  }
}
