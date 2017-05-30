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

import static com.google.common.truth.Truth.assertThat;

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
  public final void createList() throws Exception  {
    list = new LongArrayList();
  }

  @Test
  public void testAdds() throws Exception {
    for (int i = 0; i < 50; i++) {
      list.add(i);
    }
    for (int i = 0; i < 50; i++) {
      assertThat(i).isEqualTo(list.get(i));
    }
    list.add(25, 42);
    assertThat(list.get(25)).isEqualTo(42);
    assertThat(list.get(26)).isEqualTo(25);
    assertThat(list.get(list.size() - 1)).isEqualTo(49);
    assertThat(list.size()).isEqualTo(51);
    assertThat(list.indexOf(23)).isEqualTo(23);
    assertThat(list.indexOf(28)).isEqualTo(29);
  }

  @Test
  public void testAddAlls() throws Exception {
    list.addAll(new long[] {1, 2, 3, 4, 5}, 1, 3);
    assertThat(list.get(0)).isEqualTo(2);
    assertThat(list.get(1)).isEqualTo(3);
    assertThat(list.get(2)).isEqualTo(4);
    assertThat(list.size()).isEqualTo(3);
    list.addAll(new long[] {42, 41}, 0, 2, 1);
    assertThat(list.get(1)).isEqualTo(42);
    assertThat(list.get(2)).isEqualTo(41);
    assertThat(list.get(3)).isEqualTo(3);
    assertThat(list.get(4)).isEqualTo(4);
    assertThat(list.size()).isEqualTo(5);
    LongArrayList other = new LongArrayList(new long[] {5, 6, 7});
    list.addAll(other, list.size());
    assertThat(list.get(1)).isEqualTo(42);
    assertThat(list.get(4)).isEqualTo(4);
    assertThat(list.get(5)).isEqualTo(5);
    assertThat(list.get(6)).isEqualTo(6);
    assertThat(list.get(7)).isEqualTo(7);
    assertThat(list.size()).isEqualTo(8);
    list.addAll(new LongArrayList());
    assertThat(list.size()).isEqualTo(8);
    list.addAll(new long[] {});
    assertThat(list.size()).isEqualTo(8);
  }

  @Test
  public void testSet() throws Exception {
    list.addAll(new long[] {1, 2, 3});
    list.set(1, 42);
    assertThat(list.get(1)).isEqualTo(42);
    assertThat(list.size()).isEqualTo(3);
  }

  @Test
  public void testSort() throws Exception {
    list = new LongArrayList(new long[] {3, 2, 1});
    list.sort();
    assertThat(list.get(0)).isEqualTo(1);
    assertThat(list.get(1)).isEqualTo(2);
    assertThat(list.get(2)).isEqualTo(3);
    list.addAll(new long[] {-5, -2});
    list.sort(2, 5);
    assertThat(list.get(2)).isEqualTo(-5);
    assertThat(list.get(3)).isEqualTo(-2);
    assertThat(list.get(4)).isEqualTo(3);
  }

  @Test
  public void testRemoveByIndex() throws Exception {
    int last = 32;
    for (int i = 0; i <= last; i++) {
      list.add(i);
    }
    long removed = list.remove(last);
    assertThat(removed).isEqualTo(last);
    assertThat(list.size()).isEqualTo(last);
    removed = list.remove(0);
    assertThat(removed).isEqualTo(0);
    assertThat(list.get(0)).isEqualTo(1);
    assertThat(list.get(last - 2)).isEqualTo(last - 1);
    assertThat(list.size()).isEqualTo(last - 1);
  }

  @Test
  public void testRemoveByValue() throws Exception {
    int last = 19;
    for (int i = 0; i <= last; i++) {
      list.add(i);
    }
    boolean removed = list.remove((long) last);
    assertThat(removed).isTrue();
    assertThat(list.size()).isEqualTo(last);
    assertThat(list.get(last - 1)).isEqualTo(last - 1);
    removed = list.remove(3L);
    assertThat(removed).isTrue();
    assertThat(list.get(0)).isEqualTo(0);
    assertThat(list.get(last - 2)).isEqualTo(last - 1);
    assertThat(list.size()).isEqualTo(last - 1);
    removed = list.remove(42L);
    assertThat(removed).isFalse();
    assertThat(list.size()).isEqualTo(last - 1);
  }

  @Test
  public void testEnsureCapacity() throws Exception {
    int last = 65;
    for (int i = 0; i <= last; i++) {
      list.add(i);
    }
    list.ensureCapacity(512);
    assertThat(list.size()).isEqualTo(last + 1);
    assertThat(list.get(0)).isEqualTo(0);
    assertThat(list.get(last)).isEqualTo(last);
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
