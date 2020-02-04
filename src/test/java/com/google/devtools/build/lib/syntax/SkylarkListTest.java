// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.ArrayList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Sequence. */
@RunWith(JUnit4.class)
public final class SkylarkListTest extends EvaluationTestCase {

  @Test
  public void testIndex() throws Exception {
    exec("l = [1, '2', 3]");
    assertThat(eval("l[0]")).isEqualTo(1);
    assertThat(eval("l[1]")).isEqualTo("2");
    assertThat(eval("l[2]")).isEqualTo(3);

    exec("t = (1, '2', 3)");
    assertThat(eval("t[0]")).isEqualTo(1);
    assertThat(eval("t[1]")).isEqualTo("2");
    assertThat(eval("t[2]")).isEqualTo(3);
  }

  @Test
  public void testIndexOutOfBounds() throws Exception {
    checkEvalError(
        "index out of range (index is 3, but sequence has 3 elements)", "['a', 'b', 'c'][3]");
    checkEvalError(
        "index out of range (index is 10, but sequence has 3 elements)", "['a', 'b', 'c'][10]");
    checkEvalError("index out of range (index is 0, but sequence has 0 elements)", "[][0]");
  }

  @Test
  public void testNegativeIndices() throws Exception {
    exec("l = ['a', 'b', 'c']");
    assertThat(eval("l[0]")).isEqualTo("a");
    assertThat(eval("l[-1]")).isEqualTo("c");
    assertThat(eval("l[-2]")).isEqualTo("b");
    assertThat(eval("l[-3]")).isEqualTo("a");
    checkEvalError("index out of range (index is -4, but sequence has 3 elements)", "l[-4]");
    checkEvalError("index out of range (index is -1, but sequence has 0 elements)", "[][-1]");
  }

  @SuppressWarnings("unchecked")
  private Sequence<Object> listEval(String... input) throws Exception {
    return (Sequence<Object>) eval(input);
  }

  @Test
  public void testSlice() throws Exception {
    exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[0:3]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[0:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[0:1]")).containsExactly("a").inOrder();
    assertThat(listEval("l[0:0]")).isEmpty();
    assertThat(listEval("l[1:3]")).containsExactly("b", "c").inOrder();
    assertThat(listEval("l[2:3]")).containsExactly("c").inOrder();
    assertThat(listEval("l[3:3]")).isEmpty();
    assertThat(listEval("l[2:1]")).isEmpty();
    assertThat(listEval("l[3:0]")).isEmpty();

    exec("t = ('a', 'b', 'c')");
    assertThat(listEval("t[0:3]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("t[1:2]")).containsExactly("b").inOrder();
  }

  @Test
  public void testSliceDefault() throws Exception {
    exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[:]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[2:]")).containsExactly("c").inOrder();
  }

  @Test
  public void testSliceNegative() throws Exception {
    exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[-2:-1]")).containsExactly("b").inOrder();
    assertThat(listEval("l[-2:]")).containsExactly("b", "c").inOrder();
    assertThat(listEval("l[0:-1]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[-1:1]")).isEmpty();
  }

  @Test
  public void testSliceBounds() throws Exception {
    exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[0:5]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[-10:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[3:10]")).isEmpty();
    assertThat(listEval("l[-10:-9]")).isEmpty();
  }

  @Test
  public void testSliceSkip() throws Exception {
    exec("l = ['a', 'b', 'c', 'd', 'e', 'f', 'g']");
    assertThat(listEval("l[0:6:2]")).containsExactly("a", "c", "e").inOrder();
    assertThat(listEval("l[0:7:2]")).containsExactly("a", "c", "e", "g").inOrder();
    assertThat(listEval("l[0:10:2]")).containsExactly("a", "c", "e", "g").inOrder();
    assertThat(listEval("l[-6:10:2]")).containsExactly("b", "d", "f").inOrder();
    assertThat(listEval("l[1:5:3]")).containsExactly("b", "e").inOrder();
    assertThat(listEval("l[-10:3:2]")).containsExactly("a", "c").inOrder();
    assertThat(listEval("l[-10:10:1]")).containsExactly(
        "a", "b", "c", "d", "e", "f", "g").inOrder();
  }

  @Test
  public void testSliceNegativeSkip() throws Exception {
    exec("l = ['a', 'b', 'c', 'd', 'e', 'f', 'g']");
    assertThat(listEval("l[5:2:-1]")).containsExactly("f", "e", "d").inOrder();
    assertThat(listEval("l[5:2:-2]")).containsExactly("f", "d").inOrder();
    assertThat(listEval("l[5:3:-2]")).containsExactly("f").inOrder();
    assertThat(listEval("l[6::-4]")).containsExactly("g", "c").inOrder();
    assertThat(listEval("l[7::-4]")).containsExactly("g", "c").inOrder();
    assertThat(listEval("l[-1::-4]")).containsExactly("g", "c").inOrder();
    assertThat(listEval("l[-1:-10:-4]")).containsExactly("g", "c").inOrder();
    assertThat(listEval("l[-1:-3:-4]")).containsExactly("g").inOrder();
    assertThat(listEval("l[2:5:-1]")).isEmpty();
    assertThat(listEval("l[-10:5:-1]")).isEmpty();
    assertThat(listEval("l[1:-8:-1]")).containsExactly("b", "a").inOrder();

    checkEvalError("slice step cannot be zero", "l[2:5:0]");
  }

  @Test
  public void testListSize() throws Exception {
    assertThat(eval("len([42, 'hello, world', []])")).isEqualTo(3);
  }

  @Test
  public void testListEmpty() throws Exception {
    assertThat(eval("8 if [1, 2, 3] else 9")).isEqualTo(8);
    assertThat(eval("8 if [] else 9")).isEqualTo(9);
  }

  @Test
  public void testListConcat() throws Exception {
    assertThat(eval("[1, 2] + [3, 4]"))
        .isEqualTo(StarlarkList.of(/*mutability=*/ null, 1, 2, 3, 4));
  }

  @Test
  public void testConcatListIndex() throws Exception {
    exec(
        "l = [1, 2] + [3, 4]", //
        "e0 = l[0]",
        "e1 = l[1]",
        "e2 = l[2]",
        "e3 = l[3]");
    assertThat(lookup("e0")).isEqualTo(1);
    assertThat(lookup("e1")).isEqualTo(2);
    assertThat(lookup("e2")).isEqualTo(3);
    assertThat(lookup("e3")).isEqualTo(4);
  }

  @Test
  public void testConcatListHierarchicalIndex() throws Exception {
    exec(
        "l = [1] + (([2] + [3, 4]) + [5])", //
        "e0 = l[0]",
        "e1 = l[1]",
        "e2 = l[2]",
        "e3 = l[3]",
        "e4 = l[4]");
    assertThat(lookup("e0")).isEqualTo(1);
    assertThat(lookup("e1")).isEqualTo(2);
    assertThat(lookup("e2")).isEqualTo(3);
    assertThat(lookup("e3")).isEqualTo(4);
    assertThat(lookup("e4")).isEqualTo(5);
  }

  @Test
  public void testConcatListSize() throws Exception {
    assertThat(eval("len([1, 2] + [3, 4])")).isEqualTo(4);
  }

  @Test
  public void testAppend() throws Exception {
    exec("l = [1, 2]");
    assertThat(Starlark.NONE).isEqualTo(eval("l.append([3, 4])"));
    assertThat(eval("[1, 2, [3, 4]]")).isEqualTo(lookup("l"));
  }

  @Test
  public void testExtend() throws Exception {
    exec("l = [1, 2]");
    assertThat(Starlark.NONE).isEqualTo(eval("l.extend([3, 4])"));
    assertThat(eval("[1, 2, 3, 4]")).isEqualTo(lookup("l"));
  }

  @Test
  public void testConcatListToString() throws Exception {
    assertThat(eval("str([1, 2] + [3, 4])")).isEqualTo("[1, 2, 3, 4]");
  }

  @Test
  public void testConcatListNotEmpty() throws Exception {
    exec("l = [1, 2] + [3, 4]", "v = 1 if l else 0");
    assertThat(lookup("v")).isEqualTo(1);
  }

  @Test
  public void testConcatListEmpty() throws Exception {
    exec("l = [] + []", "v = 1 if l else 0");
    assertThat(lookup("v")).isEqualTo(0);
  }

  @Test
  public void testListComparison() throws Exception {
    assertThat(eval("(1, 'two', [3, 4]) == (1, 'two', [3, 4])")).isEqualTo(true);
    assertThat(eval("[1, 2, 3, 4] == [1, 2] + [3, 4]")).isEqualTo(true);
    assertThat(eval("[1, 2, 3, 4] == (1, 2, 3, 4)")).isEqualTo(false);
    assertThat(eval("[1, 2] == [1, 2, 3]")).isEqualTo(false);
    assertThat(eval("[] == []")).isEqualTo(true);
    assertThat(eval("() == ()")).isEqualTo(true);
    assertThat(eval("() == (1,)")).isEqualTo(false);
    assertThat(eval("(1) == (1,)")).isEqualTo(false);
  }

  @Test
  public void testListAddWithIndex() throws Exception {
    Mutability mutability = Mutability.create("test");
    StarlarkList<String> list = StarlarkList.newList(mutability);
    Location loc = null;
    list.add("a", loc);
    list.add("b", loc);
    list.add("c", loc);
    list.add(0, "d", loc);
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"b\", \"c\"]");
    list.add(2, "e", loc);
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"c\"]");
    list.add(4, "f", loc);
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"f\", \"c\"]");
    list.add(6, "g", loc);
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"f\", \"c\", \"g\"]");
    assertThrows(ArrayIndexOutOfBoundsException.class, () -> list.add(8, "h", loc));
  }

  @Test
  public void testMutatorsCheckMutability() throws Exception {
    Mutability mutability = Mutability.create("test");
    StarlarkList<Object> list = StarlarkList.copyOf(mutability, ImmutableList.of(1, 2, 3));
    mutability.freeze();

    // The casts force selection of the Starlark add/remove methods,
    // not the disabled ones like List.add(int, Object).
    // We could enable the List method, but then it would have to
    // report failures using unchecked exceptions.
    EvalException e =
        assertThrows(EvalException.class, () -> list.add((Object) 4, (Location) null));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen object");
    e = assertThrows(EvalException.class, () -> list.add(0, (Object) 4, (Location) null));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen object");
    e =
        assertThrows(
            EvalException.class, () -> list.addAll(ImmutableList.of(4, 5, 6), (Location) null));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen object");
    e = assertThrows(EvalException.class, () -> list.remove(0, (Location) null));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen object");
    e = assertThrows(EvalException.class, () -> list.set(0, 10, (Location) null));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen object");
  }

  @Test
  public void testCannotMutateAfterShallowFreeze() throws Exception {
    Mutability mutability = Mutability.createAllowingShallowFreeze("test");
    StarlarkList<Object> list = StarlarkList.copyOf(mutability, ImmutableList.of(1, 2, 3));
    list.unsafeShallowFreeze();

    EvalException e = assertThrows(EvalException.class, () -> list.add((Object) 4, null));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen object");
  }

  @Test
  public void testCopyOfTakesCopy() throws EvalException {
    ArrayList<String> copyFrom = Lists.newArrayList("hi");
    Mutability mutability = Mutability.create("test");
    StarlarkList<String> mutableList = StarlarkList.copyOf(mutability, copyFrom);
    copyFrom.add("added1");
    mutableList.add("added2", (Location) null);

    assertThat(copyFrom).containsExactly("hi", "added1").inOrder();
    assertThat(mutableList).containsExactly("hi", "added2").inOrder();
  }

  @Test
  public void testWrapTakesOwnershipOfArray() throws EvalException {
    String[] wrapped = {"hello"};
    Mutability mutability = Mutability.create("test");
    StarlarkList<String> mutableList = StarlarkList.wrap(mutability, wrapped);

    // Big no-no, but we're proving a point.
    wrapped[0] = "goodbye";
    assertThat(mutableList).containsExactly("goodbye");
  }

  @Test
  public void testGetSkylarkType_GivesExpectedClassesForListsAndTuples() throws Exception {
    Class<?> emptyTupleClass = Tuple.empty().getClass();
    Class<?> tupleClass = Tuple.of(1, "a", "b").getClass();
    Class<?> mutableListClass =
        StarlarkList.copyOf(thread.mutability(), Tuple.of(1, 2, 3)).getClass();

    assertThat(EvalUtils.getSkylarkType(mutableListClass)).isEqualTo(StarlarkList.class);
    assertThat(EvalUtils.getSkylarkType(emptyTupleClass)).isEqualTo(Tuple.class);
    assertThat(EvalUtils.getSkylarkType(tupleClass)).isEqualTo(Tuple.class);
  }
}
