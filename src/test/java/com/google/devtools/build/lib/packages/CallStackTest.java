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

import com.google.common.collect.ImmutableList;
import com.google.common.truth.Correspondence;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.util.List;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CallStack}. */
@RunWith(JUnit4.class)
public class CallStackTest {

  /**
   * Compare {@link StarlarkThread.CallStackEntry} using string equality since (1) it doesn't
   * currently implement equals and (2) it should have a faithful string representation anyway.
   */
  private static final Correspondence<StarlarkThread.CallStackEntry, StarlarkThread.CallStackEntry>
      STACK_ENTRY_CORRESPONDENCE =
          Correspondence.from(
              (l, r) -> l.toString().equals(r.toString()), "String-representations equal");

  @Test
  public void testCreateFromEmptyCallStack() {
    CallStack.Factory factory = new CallStack.Factory();
    CallStack result = factory.createFrom(ImmutableList.of());

    assertThat(result.size()).isEqualTo(0);
    assertThat(result.toList()).isEmpty();
  }

  @Test
  public void testCreateFromSimpleCallStack() {
    CallStack.Factory factory = new CallStack.Factory();

    ImmutableList<StarlarkThread.CallStackEntry> stack =
        ImmutableList.of(
            entryFromNameAndLocation("func1", "file1.bzl", 10, 20),
            entryFromNameAndLocation("func2", "file2.bzl", 20, 30));

    assertCallStackContents(factory.createFrom(stack), stack);
  }

  @Test
  public void testCreateFromCallStackWithLoops() {
    CallStack.Factory factory = new CallStack.Factory();

    StarlarkThread.CallStackEntry loopEntry1 =
        entryFromNameAndLocation("loop1", "file1.bzl", 10, 20);
    StarlarkThread.CallStackEntry loopEntry2 =
        entryFromNameAndLocation("loop2", "file2.bzl", 20, 30);

    ImmutableList<StarlarkThread.CallStackEntry> stack =
        ImmutableList.of(loopEntry1, loopEntry2, loopEntry1, loopEntry2);

    assertCallStackContents(factory.createFrom(stack), stack);
  }

  @Test
  public void testCreateFromConsecutiveCalls() {
    CallStack.Factory factory = new CallStack.Factory();

    ImmutableList.Builder<StarlarkThread.CallStackEntry> stackBuilder =
        ImmutableList.<StarlarkThread.CallStackEntry>builder()
            .add(entryFromNameAndLocation("f1", "f.bzl", 1, 2))
            .add(entryFromNameAndLocation("g1", "g.bzl", 2, 3));
    ImmutableList<StarlarkThread.CallStackEntry> stack1 = stackBuilder.build();
    ImmutableList<StarlarkThread.CallStackEntry> stack2 =
        stackBuilder.add(entryFromNameAndLocation("h1", "h.bzl", 3, 4)).build();

    assertCallStackContents(factory.createFrom(stack1), stack1);
    assertCallStackContents(factory.createFrom(stack2), stack2);
  }

  @Test
  public void callStackFactory_tailOptimisation() {
    CallStack.Factory factory = new CallStack.Factory();
    ImmutableList<StarlarkThread.CallStackEntry> stack1 =
        ImmutableList.of(
            entryFromNameAndLocation("target1", "a/BUILD", 1, 2),
            entryFromNameAndLocation("java_library_macro", "java_library_macro.bzl", 2, 3),
            entryFromNameAndLocation("java_library", "java_library.bzl", 4, 5));
    ImmutableList<StarlarkThread.CallStackEntry> stack2 =
        ImmutableList.of(
            entryFromNameAndLocation("target2", "b/BUILD", 6, 7),
            entryFromNameAndLocation("java_library_macro", "java_library_macro.bzl", 2, 3),
            entryFromNameAndLocation("java_library", "java_library.bzl", 4, 5));

    CallStack optimisedStack1 = factory.createFrom(stack1);
    CallStack optimisedStack2 = factory.createFrom(stack2);

    assertCallStackContents(optimisedStack1, stack1);
    assertCallStackContents(optimisedStack2, stack2);
    assertThat(optimisedStack1.head.child).isSameInstanceAs(optimisedStack2.head.child);
    assertThat(optimisedStack1.head.child.child).isSameInstanceAs(optimisedStack2.head.child.child);
  }

  @Test
  public void testSerialization() throws Exception {
    CallStack.Factory factory = new CallStack.Factory();

    ImmutableList<StarlarkThread.CallStackEntry> stackEntries1 =
        ImmutableList.of(
            entryFromNameAndLocation("somename", "f1.bzl", 1, 2),
            entryFromNameAndLocation("someOtherName", "f2.bzl", 2, 4),
            entryFromNameAndLocation("somename", "f1.bzl", 4, 2),
            entryFromNameAndLocation("somethingElse", "f3.bzl", 5, 6));

    ImmutableList<StarlarkThread.CallStackEntry> stackEntries2 =
        ImmutableList.of(entryFromNameAndLocation("shortStack", "short.bzl", 9, 10));

    CallStack callStack1 = factory.createFrom(stackEntries1);
    CallStack callStack2 = factory.createFrom(stackEntries2);

    CallStack.Serializer serializer = new CallStack.Serializer();
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    serializer.prepareCallStack(callStack1);
    serializer.prepareCallStack(callStack2);
    serializer.serializeCallStack(callStack1, codedOut);
    serializer.serializeCallStack(callStack2, codedOut);
    serializer.serializeCallStack(callStack1, codedOut);
    codedOut.flush();

    CallStack.Deserializer deserializer = new CallStack.Deserializer();
    CodedInputStream codedIn = CodedInputStream.newInstance(bytesOut.toByteArray());

    CallStack deserializedCallStack1 = deserializer.deserializeCallStack(codedIn);
    assertCallStackContents(deserializedCallStack1, stackEntries1);

    CallStack deserializedCallStack2 = deserializer.deserializeCallStack(codedIn);
    assertCallStackContents(deserializedCallStack2, stackEntries2);

    CallStack deserializedCallStack1Again = deserializer.deserializeCallStack(codedIn);
    assertCallStackContents(deserializedCallStack1Again, stackEntries1);
  }

  /** Asserts the provided {@link CallStack} faithfully represents the expected stack. */
  private static void assertCallStackContents(CallStack result, List<CallStackEntry> expected) {
    assertThat(result.size()).isEqualTo(expected.size());
    assertThat(result.toList())
        .comparingElementsUsing(STACK_ENTRY_CORRESPONDENCE)
        .containsExactlyElementsIn(expected)
        .inOrder();
    // toList and getFrame use different code paths, make sure they agree.
    for (int i = 0; i < expected.size(); i++) {
      assertThat(result.getFrame(i).toString()).isEqualTo(expected.get(i).toString());
    }
  }

  private static StarlarkThread.CallStackEntry entryFromNameAndLocation(
      String name, String file, int line, int col) {
    return new CallStackEntry(name, Location.fromFileLineColumn(file, line, col));
  }
}
