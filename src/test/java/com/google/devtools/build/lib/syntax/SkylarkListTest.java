// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Iterator;

/**
 * Tests for SkylarkList.
 */
@RunWith(JUnit4.class)
public class SkylarkListTest extends EvaluationTestCase {

  @Immutable
  private static final class CustomIterable implements Iterable<Object> {

    @Override
    public Iterator<Object> iterator() {
      // Throw an exception whenever we request the iterator, to test that lazy lists
      // are truly lazy.
      throw new IllegalArgumentException("Iterator requested");
    }
  }

  private static final SkylarkList list =
      SkylarkList.lazyList(new CustomIterable(), Integer.class);

  @Override
  public EvaluationContext newEvaluationContext() throws Exception {
    return super.newEvaluationContext().update("lazy", list);
  }

  @Test
  public void testLazyListIndex() throws Exception {
    checkEvalError("Illegal argument in call to $index: Iterator requested", "a = lazy[0]");
  }

  @Test
  public void testLazyListSize() throws Exception {
    checkEvalError("Illegal argument in call to len: Iterator requested", "a = len(lazy)");
  }

  @Test
  public void testLazyListEmpty() throws Exception {
    checkEvalError("Iterator requested", "if lazy:\n  a = 1");
  }

  @Test
  public void testLazyListConcat() throws Exception {
    eval("v = [1, 2] + lazy");
    assertThat(lookup("v")).isInstanceOf(SkylarkList.class);
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
    eval("l = [1, 2] + [3, 4]",
         "s = len(l)");
    assertEquals(4, lookup("s"));
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
