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
package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Sequence. */
@RunWith(JUnit4.class)
public final class StarlarkListTest {

  private final EvaluationTestCase ev = new EvaluationTestCase();

  @Test
  public void testIndex() throws Exception {
    ev.exec("l = [1, '2', 3]");
    assertThat(ev.eval("l[0]")).isEqualTo(StarlarkInt.of(1));
    assertThat(ev.eval("l[1]")).isEqualTo("2");
    assertThat(ev.eval("l[2]")).isEqualTo(StarlarkInt.of(3));

    ev.exec("t = (1, '2', 3)");
    assertThat(ev.eval("t[0]")).isEqualTo(StarlarkInt.of(1));
    assertThat(ev.eval("t[1]")).isEqualTo("2");
    assertThat(ev.eval("t[2]")).isEqualTo(StarlarkInt.of(3));
  }

  @Test
  public void testIndexOutOfBounds() throws Exception {
    ev.checkEvalError(
        "index out of range (index is 3, but sequence has 3 elements)", "['a', 'b', 'c'][3]");
    ev.checkEvalError(
        "index out of range (index is 10, but sequence has 3 elements)", "['a', 'b', 'c'][10]");
    ev.checkEvalError("index out of range (index is 0, but sequence has 0 elements)", "[][0]");
  }

  @Test
  public void testNegativeIndices() throws Exception {
    ev.exec("l = ['a', 'b', 'c']");
    assertThat(ev.eval("l[0]")).isEqualTo("a");
    assertThat(ev.eval("l[-1]")).isEqualTo("c");
    assertThat(ev.eval("l[-2]")).isEqualTo("b");
    assertThat(ev.eval("l[-3]")).isEqualTo("a");
    ev.checkEvalError("index out of range (index is -4, but sequence has 3 elements)", "l[-4]");
    ev.checkEvalError("index out of range (index is -1, but sequence has 0 elements)", "[][-1]");
  }

  @SuppressWarnings("unchecked")
  private Sequence<Object> listEval(String... input) throws Exception {
    return (Sequence<Object>) ev.eval(input);
  }

  @Test
  public void testSlice() throws Exception {
    ev.exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[0:3]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[0:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[0:1]")).containsExactly("a").inOrder();
    assertThat(listEval("l[0:0]")).isEmpty();
    assertThat(listEval("l[1:3]")).containsExactly("b", "c").inOrder();
    assertThat(listEval("l[2:3]")).containsExactly("c").inOrder();
    assertThat(listEval("l[3:3]")).isEmpty();
    assertThat(listEval("l[2:1]")).isEmpty();
    assertThat(listEval("l[3:0]")).isEmpty();

    ev.exec("t = ('a', 'b', 'c')");
    assertThat(listEval("t[0:3]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("t[1:2]")).containsExactly("b").inOrder();
  }

  @Test
  public void testSliceDefault() throws Exception {
    ev.exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[:]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[2:]")).containsExactly("c").inOrder();
  }

  @Test
  public void testSliceNegative() throws Exception {
    ev.exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[-2:-1]")).containsExactly("b").inOrder();
    assertThat(listEval("l[-2:]")).containsExactly("b", "c").inOrder();
    assertThat(listEval("l[0:-1]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[-1:1]")).isEmpty();
  }

  @Test
  public void testSliceBounds() throws Exception {
    ev.exec("l = ['a', 'b', 'c']");
    assertThat(listEval("l[0:5]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[-10:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[3:10]")).isEmpty();
    assertThat(listEval("l[-10:-9]")).isEmpty();
  }

  @Test
  public void testSliceSkip() throws Exception {
    ev.exec("l = ['a', 'b', 'c', 'd', 'e', 'f', 'g']");
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
    ev.exec("l = ['a', 'b', 'c', 'd', 'e', 'f', 'g']");
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

    ev.checkEvalError("slice step cannot be zero", "l[2:5:0]");
  }

  @Test
  public void testListSize() throws Exception {
    assertThat(ev.eval("len([42, 'hello, world', []])")).isEqualTo(StarlarkInt.of(3));
  }

  @Test
  public void testListEmpty() throws Exception {
    assertThat(ev.eval("8 if [1, 2, 3] else 9")).isEqualTo(StarlarkInt.of(8));
    assertThat(ev.eval("8 if [] else 9")).isEqualTo(StarlarkInt.of(9));
  }

  @Test
  public void testListConcat() throws Exception {
    assertThat(ev.eval("[1, 2] + [3, 4]"))
        .isEqualTo(
            StarlarkList.of(
                /*mutability=*/ null,
                StarlarkInt.of(1),
                StarlarkInt.of(2),
                StarlarkInt.of(3),
                StarlarkInt.of(4)));
  }

  @Test
  public void testConcatListIndex() throws Exception {
    ev.exec(
        "l = [1, 2] + [3, 4]", //
        "e0 = l[0]",
        "e1 = l[1]",
        "e2 = l[2]",
        "e3 = l[3]");
    assertThat(ev.lookup("e0")).isEqualTo(StarlarkInt.of(1));
    assertThat(ev.lookup("e1")).isEqualTo(StarlarkInt.of(2));
    assertThat(ev.lookup("e2")).isEqualTo(StarlarkInt.of(3));
    assertThat(ev.lookup("e3")).isEqualTo(StarlarkInt.of(4));
  }

  @Test
  public void testConcatListHierarchicalIndex() throws Exception {
    ev.exec(
        "l = [1] + (([2] + [3, 4]) + [5])", //
        "e0 = l[0]",
        "e1 = l[1]",
        "e2 = l[2]",
        "e3 = l[3]",
        "e4 = l[4]");
    assertThat(ev.lookup("e0")).isEqualTo(StarlarkInt.of(1));
    assertThat(ev.lookup("e1")).isEqualTo(StarlarkInt.of(2));
    assertThat(ev.lookup("e2")).isEqualTo(StarlarkInt.of(3));
    assertThat(ev.lookup("e3")).isEqualTo(StarlarkInt.of(4));
    assertThat(ev.lookup("e4")).isEqualTo(StarlarkInt.of(5));
  }

  @Test
  public void testConcatListSize() throws Exception {
    assertThat(ev.eval("len([1, 2] + [3, 4])")).isEqualTo(StarlarkInt.of(4));
  }

  @Test
  public void testAppend() throws Exception {
    ev.exec("l = [1, 2]");
    assertThat(Starlark.NONE).isEqualTo(ev.eval("l.append([3, 4])"));
    assertThat(ev.eval("[1, 2, [3, 4]]")).isEqualTo(ev.lookup("l"));
  }

  @Test
  public void testExtend() throws Exception {
    ev.exec("l = [1, 2]");
    assertThat(Starlark.NONE).isEqualTo(ev.eval("l.extend([3, 4])"));
    assertThat(ev.eval("[1, 2, 3, 4]")).isEqualTo(ev.lookup("l"));
  }

  @Test
  public void listAfterRemoveHasExpectedEqualsAndHashCode() throws Exception {
    ev.exec("l = [1, 2, 3]");
    ev.exec("l.remove(3)");
    assertThat(ev.lookup("l")).isEqualTo(ev.eval("[1, 2]"));
    assertThat(ev.lookup("l").hashCode()).isEqualTo(ev.eval("[1, 2]").hashCode());
  }

  @Test
  public void testConcatListToString() throws Exception {
    assertThat(ev.eval("str([1, 2] + [3, 4])")).isEqualTo("[1, 2, 3, 4]");
  }

  @Test
  public void testConcatListNotEmpty() throws Exception {
    ev.exec("l = [1, 2] + [3, 4]", "v = 1 if l else 0");
    assertThat(ev.lookup("v")).isEqualTo(StarlarkInt.of(1));
  }

  @Test
  public void testConcatListEmpty() throws Exception {
    ev.exec("l = [] + []", "v = 1 if l else 0");
    assertThat(ev.lookup("v")).isEqualTo(StarlarkInt.of(0));
  }

  @Test
  public void testListComparison() throws Exception {
    assertThat(ev.eval("(1, 'two', [3, 4]) == (1, 'two', [3, 4])")).isEqualTo(true);
    assertThat(ev.eval("[1, 2, 3, 4] == [1, 2] + [3, 4]")).isEqualTo(true);
    assertThat(ev.eval("[1, 2, 3, 4] == (1, 2, 3, 4)")).isEqualTo(false);
    assertThat(ev.eval("[1, 2] == [1, 2, 3]")).isEqualTo(false);
    assertThat(ev.eval("[] == []")).isEqualTo(true);
    assertThat(ev.eval("() == ()")).isEqualTo(true);
    assertThat(ev.eval("() == (1,)")).isEqualTo(false);
    assertThat(ev.eval("(1) == (1,)")).isEqualTo(false);
  }

  @Test
  public void testListAddWithIndex() throws Exception {
    Mutability mutability = Mutability.create("test");
    StarlarkList<String> list = StarlarkList.newList(mutability);
    list.addElement("a");
    list.addElement("b");
    list.addElement("c");
    list.addElementAt(0, "d");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"b\", \"c\"]");
    list.addElementAt(2, "e");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"c\"]");
    list.addElementAt(4, "f");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"f\", \"c\"]");
    list.addElementAt(6, "g");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"f\", \"c\", \"g\"]");
    assertThrows(ArrayIndexOutOfBoundsException.class, () -> list.addElementAt(8, "h"));
  }

  @Test
  public void testMutatorsCheckMutability() throws Exception {
    Mutability mutability = Mutability.create("test");
    StarlarkList<Object> list =
        StarlarkList.copyOf(
            mutability, ImmutableList.of(StarlarkInt.of(1), StarlarkInt.of(2), StarlarkInt.of(3)));
    mutability.freeze();

    EvalException e = assertThrows(EvalException.class, () -> list.addElement(StarlarkInt.of(4)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e = assertThrows(EvalException.class, () -> list.addElementAt(0, StarlarkInt.of(4)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e =
        assertThrows(
            EvalException.class,
            () ->
                list.addElements(
                    ImmutableList.of(StarlarkInt.of(4), StarlarkInt.of(5), StarlarkInt.of(6))));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e = assertThrows(EvalException.class, () -> list.removeElementAt(0));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e = assertThrows(EvalException.class, () -> list.setElementAt(0, StarlarkInt.of(10)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
  }

  @Test
  public void testCannotMutateAfterShallowFreeze() throws Exception {
    Mutability mutability = Mutability.createAllowingShallowFreeze("test");
    StarlarkList<Object> list =
        StarlarkList.copyOf(
            mutability, ImmutableList.of(StarlarkInt.of(1), StarlarkInt.of(2), StarlarkInt.of(3)));
    list.unsafeShallowFreeze();

    EvalException e = assertThrows(EvalException.class, () -> list.addElement(StarlarkInt.of(4)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
  }

  @Test
  public void testCopyOfTakesCopy() throws EvalException {
    ArrayList<String> copyFrom = Lists.newArrayList("hi");
    Mutability mutability = Mutability.create("test");
    StarlarkList<String> mutableList = StarlarkList.copyOf(mutability, copyFrom);
    copyFrom.add("added1");
    mutableList.addElement("added2");

    assertThat(copyFrom).containsExactly("hi", "added1").inOrder();
    assertThat((List<String>) mutableList).containsExactly("hi", "added2").inOrder();
  }

  @Test
  public void testWrapTakesOwnershipOfArray() throws EvalException {
    Object[] wrapped = {"hello"};
    Mutability mutability = Mutability.create("test");
    StarlarkList<Object> mutableList = StarlarkList.wrap(mutability, wrapped);

    // Big no-no, but we're proving a point.
    wrapped[0] = "goodbye";
    assertThat((List<?>) mutableList).containsExactly("goodbye");
  }

  @Test
  public void testOfReturnsListWhoseArrayElementTypeIsObject() throws EvalException {
    Mutability mu = Mutability.create("test");
    StarlarkList<Object> list = StarlarkList.of(mu, "a", "b");
    list.addElement(StarlarkInt.of(1)); // no ArrayStoreException
    assertThat(list.toString()).isEqualTo("[\"a\", \"b\", 1]");
  }
}
