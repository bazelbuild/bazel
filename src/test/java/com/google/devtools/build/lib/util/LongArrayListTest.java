// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link LongArrayList}.
 */
@RunWith(JUnit4.class)
public class LongArrayListTest {

  private LongArrayList list;

  @Before
  public void setUp() {
    list = new LongArrayList();
  }

  @Test
  public void testAdds() throws Exception {
    for (int i = 0; i < 50; i++) {
      list.add(i);
    }
    for (int i = 0; i < 50; i++) {
      assertEquals(list.get(i), i);
    }
    list.add(25, 42);
    assertEquals(42, list.get(25));
    assertEquals(25, list.get(26));
    assertEquals(49, list.get(list.size() - 1));
    assertEquals(51, list.size());
    assertEquals(23, list.indexOf(23));
    assertEquals(29, list.indexOf(28));
  }

  @Test
  public void testAddAlls() throws Exception {
    list.addAll(new long[] {1, 2, 3, 4, 5}, 1, 3);
    assertEquals(2, list.get(0));
    assertEquals(3, list.get(1));
    assertEquals(4, list.get(2));
    assertEquals(3, list.size());
    list.addAll(new long[] {42, 41}, 0, 2, 1);
    assertEquals(42, list.get(1));
    assertEquals(41, list.get(2));
    assertEquals(3, list.get(3));
    assertEquals(4, list.get(4));
    assertEquals(5, list.size());
    LongArrayList other = new LongArrayList(new long[] {5, 6, 7});
    list.addAll(other, list.size());
    assertEquals(42, list.get(1));
    assertEquals(4, list.get(4));
    assertEquals(5, list.get(5));
    assertEquals(6, list.get(6));
    assertEquals(7, list.get(7));
    assertEquals(8, list.size());
    list.addAll(new LongArrayList());
    assertEquals(8, list.size());
    list.addAll(new long[] {});
    assertEquals(8, list.size());
  }

  @Test
  public void testSet() throws Exception {
    list.addAll(new long[] {1, 2, 3});
    list.set(1, 42);
    assertEquals(42, list.get(1));
    assertEquals(3, list.size());
  }

  @Test
  public void testSort() throws Exception {
    list = new LongArrayList(new long[] {3, 2, 1});
    list.sort();
    assertEquals(1, list.get(0));
    assertEquals(2, list.get(1));
    assertEquals(3, list.get(2));
    list.addAll(new long[] {-5, -2});
    list.sort(2, 5);
    assertEquals(-5, list.get(2));
    assertEquals(-2, list.get(3));
    assertEquals(3, list.get(4));
  }

  @Test
  public void testRemoveByIndex() throws Exception {
    int last = 32;
    for (int i = 0; i <= last; i++) {
      list.add(i);
    }
    long removed = list.remove(last);
    assertEquals(last, removed);
    assertEquals(last, list.size());
    removed = list.remove(0);
    assertEquals(0, removed);
    assertEquals(1, list.get(0));
    assertEquals(last - 1, list.get(last - 2));
    assertEquals(last - 1, list.size());
  }

  @Test
  public void testRemoveByValue() throws Exception {
    int last = 19;
    for (int i = 0; i <= last; i++) {
      list.add(i);
    }
    boolean removed = list.remove((long) last);
    assertTrue(removed);
    assertEquals(last, list.size());
    assertEquals(last - 1, list.get(last - 1));
    removed = list.remove(3L);
    assertTrue(removed);
    assertEquals(0, list.get(0));
    assertEquals(last - 1, list.get(last - 2));
    assertEquals(last - 1, list.size());
    removed = list.remove(42L);
    assertFalse(removed);
    assertEquals(last - 1, list.size());
  }

  @Test
  public void testEnsureCapacity() throws Exception {
    int last = 65;
    for (int i = 0; i <= last; i++) {
      list.add(i);
    }
    list.ensureCapacity(512);
    assertEquals(last + 1, list.size());
    assertEquals(0, list.get(0));
    assertEquals(last, list.get(last));
  }

  @Test(expected = IndexOutOfBoundsException.class)
  public void testRemoveExceptionEmpty() throws Exception {
    list.remove(0);
  }

  @Test(expected = IndexOutOfBoundsException.class)
  public void testRemoveExceptionFilled() throws Exception {
    for (int i = 0; i < 15; i++) {
      list.add(i);
    }
    list.remove(15);
  }

  @Test(expected = IndexOutOfBoundsException.class)
  public void testGetException() throws Exception {
    for (int i = 0; i < 15; i++) {
      list.add(i);
    }
    list.get(15);
  }
}
