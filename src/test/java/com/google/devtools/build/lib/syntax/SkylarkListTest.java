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
import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkList.
 */
@RunWith(JUnit4.class)
public class SkylarkListTest extends EvaluationTestCase {

  @Test
  public void testIndex() throws Exception {
    eval("l = [1, '2', 3]");
    assertThat(eval("l[0]")).isEqualTo(1);
    assertThat(eval("l[1]")).isEqualTo("2");
    assertThat(eval("l[2]")).isEqualTo(3);

    eval("t = (1, '2', 3)");
    assertThat(eval("t[0]")).isEqualTo(1);
    assertThat(eval("t[1]")).isEqualTo("2");
    assertThat(eval("t[2]")).isEqualTo(3);
  }

  @Test
  public void testIndexOutOfBounds() throws Exception {
    checkEvalError("Index out of range (index is 3, but sequence has 3 elements)",
        "['a', 'b', 'c'][3]");
    checkEvalError("Index out of range (index is 10, but sequence has 3 elements)",
        "['a', 'b', 'c'][10]");
    checkEvalError("Index out of range (index is 0, but sequence has 0 elements)",
        "[][0]");
  }

  @Test
  public void testNegativeIndices() throws Exception {
    eval("l = ['a', 'b', 'c']");
    assertThat(eval("l[0]")).isEqualTo("a");
    assertThat(eval("l[-1]")).isEqualTo("c");
    assertThat(eval("l[-2]")).isEqualTo("b");
    assertThat(eval("l[-3]")).isEqualTo("a");
    checkEvalError("Index out of range (index is -4, but sequence has 3 elements)",
        "l[-4]");
    checkEvalError("Index out of range (index is -1, but sequence has 0 elements)",
        "[][-1]");
  }

  @SuppressWarnings("unchecked")
  private SkylarkList<Object> listEval(String... input) throws Exception {
    return (SkylarkList<Object>) eval(input);
  }

  @Test
  public void testSlice() throws Exception {
    eval("l = ['a', 'b', 'c']");
    assertThat(listEval("l[0:3]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[0:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[0:1]")).containsExactly("a").inOrder();
    assertThat(listEval("l[0:0]")).isEmpty();
    assertThat(listEval("l[1:3]")).containsExactly("b", "c").inOrder();
    assertThat(listEval("l[2:3]")).containsExactly("c").inOrder();
    assertThat(listEval("l[3:3]")).isEmpty();
    assertThat(listEval("l[2:1]")).isEmpty();
    assertThat(listEval("l[3:0]")).isEmpty();

    eval("t = ('a', 'b', 'c')");
    assertThat(listEval("t[0:3]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("t[1:2]")).containsExactly("b").inOrder();
  }

  @Test
  public void testSliceDefault() throws Exception {
    eval("l = ['a', 'b', 'c']");
    assertThat(listEval("l[:]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[2:]")).containsExactly("c").inOrder();
  }

  @Test
  public void testSliceNegative() throws Exception {
    eval("l = ['a', 'b', 'c']");
    assertThat(listEval("l[-2:-1]")).containsExactly("b").inOrder();
    assertThat(listEval("l[-2:]")).containsExactly("b", "c").inOrder();
    assertThat(listEval("l[0:-1]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[-1:1]")).isEmpty();
  }

  @Test
  public void testSliceBounds() throws Exception {
    eval("l = ['a', 'b', 'c']");
    assertThat(listEval("l[0:5]")).containsExactly("a", "b", "c").inOrder();
    assertThat(listEval("l[-10:2]")).containsExactly("a", "b").inOrder();
    assertThat(listEval("l[3:10]")).isEmpty();
    assertThat(listEval("l[-10:-9]")).isEmpty();
  }

  @Test
  public void testSliceSkip() throws Exception {
    eval("l = ['a', 'b', 'c', 'd', 'e', 'f', 'g']");
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
    eval("l = ['a', 'b', 'c', 'd', 'e', 'f', 'g']");
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

    checkEvalError("Slice step cannot be zero", "l[2:5:0]");
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
        .isEqualTo(SkylarkList.createImmutable(Tuple.of(1, 2, 3, 4)));
  }

  @Test
  public void testConcatListIndex() throws Exception {
    eval("l = [1, 2] + [3, 4]",
        "e0 = l[0]",
        "e1 = l[1]",
        "e2 = l[2]",
        "e3 = l[3]");
    assertEquals(1, lookup("e0"));
    assertEquals(2, lookup("e1"));
    assertEquals(3, lookup("e2"));
    assertEquals(4, lookup("e3"));
  }

  @Test
  public void testConcatListHierarchicalIndex() throws Exception {
    eval("l = [1] + (([2] + [3, 4]) + [5])",
         "e0 = l[0]",
         "e1 = l[1]",
         "e2 = l[2]",
         "e3 = l[3]",
         "e4 = l[4]");
    assertEquals(1, lookup("e0"));
    assertEquals(2, lookup("e1"));
    assertEquals(3, lookup("e2"));
    assertEquals(4, lookup("e3"));
    assertEquals(5, lookup("e4"));
  }

  @Test
  public void testConcatListSize() throws Exception {
    assertEquals(4, eval("len([1, 2] + [3, 4])"));
  }

  @Test
  public void testAppend() throws Exception {
    eval("l = [1, 2]");
    assertEquals(eval("l.append([3, 4])"), Runtime.NONE);
    assertEquals(lookup("l"), eval("[1, 2, [3, 4]]"));
  }

  @Test
  public void testExtend() throws Exception {
    eval("l = [1, 2]");
    assertEquals(eval("l.extend([3, 4])"), Runtime.NONE);
    assertEquals(lookup("l"), eval("[1, 2, 3, 4]"));
  }

  @Test
  public void testConcatListToString() throws Exception {
    eval("l = [1, 2] + [3, 4]",
         "s = str(l)");
    assertEquals("[1, 2, 3, 4]", lookup("s"));
  }

  @Test
  public void testConcatListNotEmpty() throws Exception {
    eval("l = [1, 2] + [3, 4]",
        "if l:",
        "  v = 1",
        "else:",
        "  v = 0");
    assertEquals(1, lookup("v"));
  }

  @Test
  public void testConcatListEmpty() throws Exception {
    eval("l = [] + []",
        "if l:",
        "  v = 1",
        "else:",
        "  v = 0");
    assertEquals(0, lookup("v"));
  }

  @Test
  public void testListComparison() throws Exception {
    assertEquals(true, eval("(1, 'two', [3, 4]) == (1, 'two', [3, 4])"));
    assertEquals(true, eval("[1, 2, 3, 4] == [1, 2] + [3, 4]"));
    assertEquals(false, eval("[1, 2, 3, 4] == (1, 2, 3, 4)"));
    assertEquals(false, eval("[1, 2] == [1, 2, 3]"));
    assertEquals(true, eval("[] == []"));
    assertEquals(true, eval("() == ()"));
    assertEquals(false, eval("() == (1,)"));
    assertEquals(false, eval("(1) == (1,)"));
  }
}
