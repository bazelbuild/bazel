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
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.MethodLibrary;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Iterator;

/**
 * Tests for SkylarkList.
 */
@RunWith(JUnit4.class)
public class SkylarkListTest extends AbstractEvaluationTestCase {

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
  private static final ImmutableMap<String, SkylarkType> extraObjects =
      ImmutableMap.of("lazy", SkylarkType.of(SkylarkList.class, Integer.class));

  private Environment env;

  @Before
  public void setUp() throws Exception {

    env = new SkylarkEnvironment(syntaxEvents.collector());
    env.update("lazy", list);
    MethodLibrary.setupMethodEnvironment(env);
  }

  @Test
  public void testLazyListIndex() throws Exception {
    checkError("Iterator requested", "a = lazy[0]");
  }

  @Test
  public void testLazyListSize() throws Exception {
    checkError("Iterator requested", "a = len(lazy)");
  }

  @Test
  public void testLazyListEmpty() throws Exception {
    checkError("Iterator requested", "if lazy:\n  a = 1");
  }

  @Test
  public void testLazyListConcat() throws Exception {
    exec("v = [1, 2] + lazy");
    assertThat(env.lookup("v")).isInstanceOf(SkylarkList.class);
  }

  @Test
  public void testConcatListIndex() throws Exception {
    exec("l = [1, 2] + [3, 4]",
         "e0 = l[0]",
         "e1 = l[1]",
         "e2 = l[2]",
         "e3 = l[3]");
    assertEquals(1, env.lookup("e0"));
    assertEquals(2, env.lookup("e1"));
    assertEquals(3, env.lookup("e2"));
    assertEquals(4, env.lookup("e3"));
  }

  @Test
  public void testConcatListHierarchicalIndex() throws Exception {
    exec("l = [1] + (([2] + [3, 4]) + [5])",
         "e0 = l[0]",
         "e1 = l[1]",
         "e2 = l[2]",
         "e3 = l[3]",
         "e4 = l[4]");
    assertEquals(1, env.lookup("e0"));
    assertEquals(2, env.lookup("e1"));
    assertEquals(3, env.lookup("e2"));
    assertEquals(4, env.lookup("e3"));
    assertEquals(5, env.lookup("e4"));
  }

  @Test
  public void testConcatListSize() throws Exception {
    exec("l = [1, 2] + [3, 4]",
         "s = len(l)");
    assertEquals(4, env.lookup("s"));
  }

  @Test
  public void testConcatListToString() throws Exception {
    exec("l = [1, 2] + [3, 4]",
         "s = str(l)");
    assertEquals("[1, 2, 3, 4]", env.lookup("s"));
  }

  @Test
  public void testConcatListNotEmpty() throws Exception {
    exec("l = [1, 2] + [3, 4]",
        "if l:",
        "  v = 1",
        "else:",
        "  v = 0");
    assertEquals(1, env.lookup("v"));
  }

  @Test
  public void testConcatListEmpty() throws Exception {
    exec("l = [] + []",
        "if l:",
        "  v = 1",
        "else:",
        "  v = 0");
    assertEquals(0, env.lookup("v"));
  }

  private void exec(String... input) throws Exception {
    exec(parseFileForSkylark(Joiner.on("\n").join(input), extraObjects), env);
  }

  private void checkError(String msg, String... input) throws Exception {
    try {
      exec(input);
      fail();
    } catch (Exception e) {
      assertThat(e).hasMessage(msg);
    }
  }
}
