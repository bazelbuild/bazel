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
