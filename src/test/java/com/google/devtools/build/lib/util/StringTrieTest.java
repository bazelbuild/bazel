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
package com.google.devtools.build.lib.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link StringTrie}.
 */
@RunWith(JUnit4.class)
public class StringTrieTest {
  @Test
  public void empty() {
    StringTrie<Integer> cut = new StringTrie<>();
    assertNull(cut.get(""));
    assertNull(cut.get("a"));
    assertNull(cut.get("ab"));
  }

  @Test
  public void simple() {
    StringTrie<Integer> cut = new StringTrie<>();
    cut.put("a", 1);
    cut.put("b", 2);

    assertNull(cut.get(""));
    assertEquals(1, cut.get("a").intValue());
    assertEquals(1, cut.get("ab").intValue());
    assertEquals(1, cut.get("abc").intValue());

    assertEquals(2, cut.get("b").intValue());
  }

  @Test
  public void ancestors() {
    StringTrie<Integer> cut = new StringTrie<>();
    cut.put("abc", 3);
    assertNull(cut.get(""));
    assertNull(cut.get("a"));
    assertNull(cut.get("ab"));
    assertEquals(3, cut.get("abc").intValue());
    assertEquals(3, cut.get("abcd").intValue());

    cut.put("a", 1);
    assertEquals(1, cut.get("a").intValue());
    assertEquals(1, cut.get("ab").intValue());
    assertEquals(3, cut.get("abc").intValue());

    cut.put("", 0);
    assertEquals(0, cut.get("").intValue());
    assertEquals(0, cut.get("b").intValue());
    assertEquals(1, cut.get("a").intValue());
  }
}
