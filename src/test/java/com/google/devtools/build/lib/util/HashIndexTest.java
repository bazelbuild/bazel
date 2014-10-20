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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.List;

/**
 *
 */
@RunWith(JUnit4.class)
public class HashIndexTest {

  @Test
  public void basics() throws Exception {
    HashIndex<String> index = new HashIndex<>();
    assertEquals(0, index.size());
    assertEquals(-1, index.indexOf("zero"));
    assertEquals(0, index.put("zero"));
    assertEquals(1, index.size());
    assertEquals(0, index.indexOf("zero"));
    assertEquals("zero", index.elementOf(0));

    assertEquals(0, index.put("zero")); // idempotent
    assertEquals(1, index.size());
    assertEquals(0, index.indexOf("zero"));
    assertEquals("zero", index.elementOf(0));

    assertEquals(1, index.put("one"));
    assertEquals(2, index.size());
    assertEquals(1, index.indexOf("one"));
    assertEquals("one", index.elementOf(1));
  }

  @Test
  public void constructionFromArray() throws Exception {
    List<String> array = Arrays.asList("zero", "one", "two");
    HashIndex<String> index = new HashIndex<>(array);

    assertEquals(3, index.size());

    // Test relational duality:
    for (String element: array) {
      assertEquals(element, array.get(index.indexOf(element)));
    }
    for (int ii = 0; ii < index.size(); ++ii) {
      assertEquals(ii, index.indexOf(array.get(ii)));
      assertEquals(array.get(ii), index.elementOf(ii));
    }
  }

  @Test
  public void constructionWithDuplicates() throws Exception {
    Index<String> index =  new HashIndex<>("zero", "one", "two", "zero");
    assertEquals(3, index.size());
  }

  @Test
  public void equals() throws Exception {
    Index<String> index1 = new HashIndex<>("one", "two");
    Index<String> index2 = new HashIndex<>();
    assertEquals(0, index2.size());
    index2.put("one");
    index2.put("two");
    assertEquals(index1, index2);
    assertEquals(index2, index1);
    assertEquals(index1.hashCode(), index2.hashCode());
  }

}
