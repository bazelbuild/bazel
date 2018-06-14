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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link DebuggerSerialization}. */
@RunWith(JUnit4.class)
public final class DebuggerSerializationTest {

  @Test
  public void testSimpleNestedSet() {
    Set<String> children = ImmutableSet.of("a", "b");
    SkylarkNestedSet set =
        SkylarkNestedSet.of(Object.class, NestedSetBuilder.stableOrder().addAll(children).build());

    Value value = DebuggerSerialization.getValueProto("name", set);

    assertTypeAndDescription(set, value);
    assertThat(value.getChildList()).hasSize(3);
    assertThat(value.getChild(0))
        .isEqualTo(
            Value.newBuilder()
                .setLabel("order")
                .setType("Traversal order")
                .setDescription("default")
                .build());
    assertEqualIgnoringTypeAndDescription(
        value.getChild(1), DebuggerSerialization.getValueProto("directs", children));
    assertEqualIgnoringTypeAndDescription(
        value.getChild(2), DebuggerSerialization.getValueProto("transitives", ImmutableList.of()));
  }

  @Test
  public void testNestedSetWithNestedChildren() {
    NestedSet<String> innerNestedSet =
        NestedSetBuilder.<String>stableOrder().add("inner1").add("inner2").build();
    ImmutableSet<String> directChildren = ImmutableSet.of("a", "b");
    SkylarkNestedSet outerSet =
        SkylarkNestedSet.of(
            String.class,
            NestedSetBuilder.<String>linkOrder()
                .addAll(directChildren)
                .addTransitive(innerNestedSet)
                .build());

    Value value = DebuggerSerialization.getValueProto("name", outerSet);

    assertTypeAndDescription(outerSet, value);
    assertThat(value.getChildList()).hasSize(3);
    assertThat(value.getChild(0))
        .isEqualTo(
            Value.newBuilder()
                .setLabel("order")
                .setType("Traversal order")
                .setDescription("topological")
                .build());
    assertEqualIgnoringTypeAndDescription(
        value.getChild(1), DebuggerSerialization.getValueProto("directs", directChildren));
    assertEqualIgnoringTypeAndDescription(
        value.getChild(2),
        DebuggerSerialization.getValueProto(
            "transitives", ImmutableList.of(new NestedSetView<>(innerNestedSet))));
  }

  @Test
  public void testSimpleMap() {
    Map<String, Integer> map = ImmutableMap.of("a", 1, "b", 2);

    Value value = DebuggerSerialization.getValueProto("name", map);

    assertTypeAndDescription(map, value);
    assertThat(value.getChildList()).hasSize(2);
    assertThat(value.getChild(0).getLabel()).isEqualTo("[0]");
    assertThat(value.getChild(0).getChildList())
        .isEqualTo(
            ImmutableList.of(
                DebuggerSerialization.getValueProto("key", "a"),
                DebuggerSerialization.getValueProto("value", 1)));
    assertThat(value.getChild(1).getLabel()).isEqualTo("[1]");
    assertThat(value.getChild(1).getChildList())
        .isEqualTo(
            ImmutableList.of(
                DebuggerSerialization.getValueProto("key", "b"),
                DebuggerSerialization.getValueProto("value", 2)));
  }

  @Test
  public void testNestedMap() {
    Set<String> set = ImmutableSet.of("a", "b");
    Map<String, Object> map = ImmutableMap.of("a", set);

    Value value = DebuggerSerialization.getValueProto("name", map);

    assertTypeAndDescription(map, value);
    assertThat(value.getChildList()).hasSize(1);
    assertThat(value.getChild(0).getLabel()).isEqualTo("[0]");
    assertThat(value.getChild(0).getChildList())
        .isEqualTo(
            ImmutableList.of(
                DebuggerSerialization.getValueProto("key", "a"),
                DebuggerSerialization.getValueProto("value", set)));
  }

  @Test
  public void testSimpleIterable() {
    Iterable<Integer> iter = ImmutableList.of(1, 2);

    Value value = DebuggerSerialization.getValueProto("name", iter);

    assertTypeAndDescription(iter, value);
    assertThat(value.getChildList()).hasSize(2);
    assertThat(value.getChild(0)).isEqualTo(DebuggerSerialization.getValueProto("[0]", 1));
    assertThat(value.getChild(1)).isEqualTo(DebuggerSerialization.getValueProto("[1]", 2));
  }

  @Test
  public void testNestedIterable() {
    Iterable<Object> iter = ImmutableList.of(ImmutableList.of(1, 2));

    Value value = DebuggerSerialization.getValueProto("name", iter);

    assertTypeAndDescription(iter, value);
    assertThat(value.getChildList()).hasSize(1);
    assertThat(value.getChild(0))
        .isEqualTo(DebuggerSerialization.getValueProto("[0]", ImmutableList.of(1, 2)));
  }

  @Test
  public void testSimpleArray() {
    int[] array = new int[] {1, 2};

    Value value = DebuggerSerialization.getValueProto("name", array);

    assertTypeAndDescription(array, value);
    assertThat(value.getChildList()).hasSize(2);
    assertThat(value.getChild(0)).isEqualTo(DebuggerSerialization.getValueProto("[0]", 1));
    assertThat(value.getChild(1)).isEqualTo(DebuggerSerialization.getValueProto("[1]", 2));
  }

  @Test
  public void testNestedArray() {
    Object[] array = new Object[] {1, ImmutableList.of(2, 3)};

    Value value = DebuggerSerialization.getValueProto("name", array);

    assertTypeAndDescription(array, value);
    assertThat(value.getChildList()).hasSize(2);
    assertThat(value.getChild(0)).isEqualTo(DebuggerSerialization.getValueProto("[0]", 1));
    assertThat(value.getChild(1))
        .isEqualTo(DebuggerSerialization.getValueProto("[1]", ImmutableList.of(2, 3)));
  }

  @Test
  public void testUnrecognizedObjectOrSkylarkPrimitiveHasNoChildren() {
    assertThat(DebuggerSerialization.getValueProto("name", 1).getChildList()).isEmpty();
    assertThat(DebuggerSerialization.getValueProto("name", "string").getChildList()).isEmpty();
    assertThat(DebuggerSerialization.getValueProto("name", new Object()).getChildList()).isEmpty();
  }

  private static void assertTypeAndDescription(Object object, Value value) {
    assertThat(value.getType()).isEqualTo(EvalUtils.getDataTypeName(object));
    assertThat(value.getDescription()).isEqualTo(Printer.repr(object));
  }

  /**
   * Type and description are implementation dependent (e.g. NestedSetView#directs returns a list
   * instead of a set if there are no duplicates, which changes both 'type' and 'description').
   */
  private static void assertEqualIgnoringTypeAndDescription(Value value1, Value value2) {
    assertThat(value1.getLabel()).isEqualTo(value2.getLabel());
    assertThat(value1.getChildCount()).isEqualTo(value2.getChildCount());
    for (int i = 0; i < value1.getChildCount(); i++) {
      assertEqualIgnoringTypeAndDescription(value1.getChild(i), value2.getChild(i));
    }
  }
}
