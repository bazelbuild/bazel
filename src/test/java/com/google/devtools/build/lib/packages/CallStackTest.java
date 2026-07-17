// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link CallStack}. */
@RunWith(JUnit4.class)
public final class CallStackTest {

  @Test
  public void emptyCallStack_null() {
    assertThat(CallStack.compact(ImmutableList.of(), 0)).isNull();
    assertThat(CallStack.compact(ImmutableList.of(), 1)).isNull();
    assertThat(CallStack.compact(ImmutableList.of(), 42)).isNull();
  }

  @Test
  public void singleFrameCallStack_nullInterior() {
    ImmutableList<StarlarkThread.CallStackEntry> stack =
        ImmutableList.of(entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "BUILD", 10, 20));

    CallStack.Node compacted = CallStack.compact(stack, 0);
    assertThat(compacted).isNotNull();
    assertThat(compacted.next()).isNull();
    assertThat(CallStack.compact(stack, 1)).isNull();
    assertThat(CallStack.compact(stack, 42)).isNull();
  }

  @Test
  public void simpleCallStack() {
    ImmutableList<StarlarkThread.CallStackEntry> stack =
        ImmutableList.of(
            entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "BUILD", 10, 20),
            entryFromNameAndLocation("func", "file.bzl", 20, 30));

    CallStack.Node compacted0 = CallStack.compact(stack, 0);
    CallStack.Node compacted1 = CallStack.compact(stack, 1);
    assertThat(compacted0.next()).isEqualTo(compacted1);
    assertThat(compacted1.next()).isNull();
    assertCallStackContents(compacted0, stack, 0);
    assertCallStackContents(compacted1, stack, 1);
    assertThat(CallStack.compact(stack, 2)).isNull();
  }

  @Test
  public void callStackWithLoops() {
    StarlarkThread.CallStackEntry loopEntry1 =
        entryFromNameAndLocation("loop1", "file1.bzl", 20, 30);
    StarlarkThread.CallStackEntry loopEntry2 =
        entryFromNameAndLocation("loop2", "file2.bzl", 30, 40);

    ImmutableList<StarlarkThread.CallStackEntry> stack =
        ImmutableList.of(
            entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "BUILD", 10, 20),
            loopEntry1,
            loopEntry2,
            loopEntry1,
            loopEntry2);

    assertCallStackContents(CallStack.compact(stack, 1), stack, 1);
  }

  @Test
  public void consecutiveCalls() {
    ImmutableList.Builder<StarlarkThread.CallStackEntry> stackBuilder =
        ImmutableList.<StarlarkThread.CallStackEntry>builder()
            .add(entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "BUILD", 1, 2))
            .add(entryFromNameAndLocation("f1", "f.bzl", 2, 3))
            .add(entryFromNameAndLocation("g1", "g.bzl", 3, 4));
    ImmutableList<StarlarkThread.CallStackEntry> stack1 = stackBuilder.build();
    ImmutableList<StarlarkThread.CallStackEntry> stack2 =
        stackBuilder.add(entryFromNameAndLocation("h1", "h.bzl", 4, 5)).build();

    assertCallStackContents(CallStack.compact(stack1, 1), stack1, 1);
    assertCallStackContents(CallStack.compact(stack2, 1), stack2, 1);
  }

  @Test
  public void sharesCommonTail() {
    ImmutableList<StarlarkThread.CallStackEntry> stack1 =
        ImmutableList.of(
            entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "a/BUILD", 1, 2),
            entryFromNameAndLocation("java_library_macro", "java_library_macro.bzl", 2, 3),
            entryFromNameAndLocation("java_library", "java_library.bzl", 4, 5));
    ImmutableList<StarlarkThread.CallStackEntry> stack2 =
        ImmutableList.of(
            entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "b/BUILD", 6, 7),
            entryFromNameAndLocation("java_library_macro", "java_library_macro.bzl", 2, 3),
            entryFromNameAndLocation("java_library", "java_library.bzl", 4, 5));

    CallStack.Node optimizedInteriorStack1 = CallStack.compact(stack1, 1);
    CallStack.Node optimizedInteriorStack2 = CallStack.compact(stack2, 1);
    CallStack.Node optimizedFullStack1 = CallStack.compact(stack1, 0);
    CallStack.Node optimizedFullStack2 = CallStack.compact(stack2, 0);

    assertCallStackContents(optimizedInteriorStack1, stack1, 1);
    assertCallStackContents(optimizedInteriorStack2, stack2, 1);
    assertCallStackContents(optimizedFullStack1, stack1, 0);
    assertCallStackContents(optimizedFullStack2, stack2, 0);
    assertThat(optimizedInteriorStack2.next().next())
        .isSameInstanceAs(optimizedInteriorStack2.next().next());
    assertThat(optimizedFullStack1.next()).isSameInstanceAs(optimizedFullStack2.next());
    assertThat(optimizedFullStack1.next()).isSameInstanceAs(optimizedInteriorStack1);
  }

  @Test
  public void serialization() throws Exception {
    ImmutableList<StarlarkThread.CallStackEntry> stackEntries1 =
        ImmutableList.of(
            entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "BUILD", 1, 2),
            entryFromNameAndLocation("somename", "f1.bzl", 1, 2),
            entryFromNameAndLocation("someOtherName", "f2.bzl", 2, 4),
            entryFromNameAndLocation("somename", "f1.bzl", 4, 2),
            entryFromNameAndLocation("somethingElse", "f3.bzl", 5, 6));

    ImmutableList<StarlarkThread.CallStackEntry> stackEntries2 =
        ImmutableList.of(entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "BUILD", 9, 10));

    CallStack.Node interiorStack1 = CallStack.compact(stackEntries1, 1);
    CallStack.Node interiorStack2 = CallStack.compact(stackEntries2, 1);
    RuleClass ruleClass = mock(RuleClass.class);
    Mockito.when(ruleClass.getAttributeProvider()).thenReturn(mock(AttributeProvider.class));
    Rule rule1 =
        new Rule(
            mock(Package.class),
            Label.parseCanonicalUnchecked("//pkg:rule1"),
            ruleClass,
            stackEntries1.get(0).location,
            interiorStack1);
    Rule rule2 =
        new Rule(
            mock(Package.class),
            Label.parseCanonicalUnchecked("//pkg:rule2"),
            ruleClass,
            stackEntries2.get(0).location,
            interiorStack2);

    SerializationContext serializer =
        new ObjectCodecs().getMemoizingSerializationContextForTesting();
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);

    serializer.serialize(CallStack.getFullCallStack(rule1), codedOut);
    serializer.serialize(CallStack.getFullCallStack(rule2), codedOut);
    serializer.serialize(CallStack.getFullCallStack(rule1), codedOut);
    codedOut.flush();

    DeserializationContext deserializer =
        new ObjectCodecs().getMemoizingDeserializationContextForTesting();
    CodedInputStream codedIn = CodedInputStream.newInstance(bytesOut.toByteArray());

    CallStack.Node deserializedCallStack1 = deserializer.deserialize(codedIn);
    assertThat(deserializedCallStack1.toLocation()).isEqualTo(rule1.getLocation());
    assertCallStackContents(deserializedCallStack1.next(), stackEntries1, 1);

    CallStack.Node deserializedCallStack2 = deserializer.deserialize(codedIn);
    assertThat(deserializedCallStack2.toLocation()).isEqualTo(rule2.getLocation());
    assertCallStackContents(deserializedCallStack2.next(), stackEntries2, 1);

    CallStack.Node deserializedCallStack1Again = deserializer.deserialize(codedIn);
    assertThat(deserializedCallStack1Again.toLocation()).isEqualTo(rule1.getLocation());
    assertCallStackContents(deserializedCallStack1Again.next(), stackEntries1, 1);
  }

  @Test
  public void concatenate() {
    ImmutableList<StarlarkThread.CallStackEntry> outerStack =
        ImmutableList.of(
            entryFromNameAndLocation(StarlarkThread.TOP_LEVEL, "BUILD", 10, 20),
            entryFromNameAndLocation("foo", "f.bzl", 1, 2),
            entryFromNameAndLocation("bar", "g.bzl", 3, 4));
    ImmutableList<StarlarkThread.CallStackEntry> innerStack =
        ImmutableList.of(
            entryFromNameAndLocation("baz", "h.bzl", 5, 6),
            entryFromNameAndLocation("qux", "g.bzl", 7, 8));

    assertThat(CallStack.concatenate(null, null)).isNull();
    assertCallStackContents(
        CallStack.concatenate(CallStack.compact(outerStack, 1), null), outerStack, 1);
    assertCallStackContents(
        CallStack.concatenate(null, CallStack.compact(innerStack, 0)), innerStack, 0);
    assertCallStackContents(
        CallStack.concatenate(CallStack.compact(outerStack, 1), CallStack.compact(innerStack, 0)),
        ImmutableList.<StarlarkThread.CallStackEntry>builder()
            .addAll(outerStack)
            .addAll(innerStack)
            .build(),
        1);
  }

  /**
   * Asserts the provided {@link CallStack.Node} faithfully represents the expected stack, ignoring
   * {@code expectedStart} of the expected stack's outer frames.
   */
  private static void assertCallStackContents(
      CallStack.Node compacted, List<StarlarkThread.CallStackEntry> expected, int expectedStart) {
    List<StarlarkThread.CallStackEntry> reconstituted = new ArrayList<>();
    for (CallStack.Node node = compacted; node != null; node = node.next()) {
      reconstituted.add(node.toCallStackEntry());
    }
    assertThat(reconstituted).isEqualTo(expected.subList(expectedStart, expected.size()));
  }

  private static StarlarkThread.CallStackEntry entryFromNameAndLocation(
      String name, String file, int line, int col) {
    return StarlarkThread.callStackEntry(name, Location.fromFileLineColumn(file, line, col));
  }
}
