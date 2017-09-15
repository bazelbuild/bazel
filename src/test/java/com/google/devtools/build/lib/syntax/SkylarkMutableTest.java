// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SkylarkMutable}. */
@RunWith(JUnit4.class)
public final class SkylarkMutableTest {

  @Test
  public void testViewsCheckMutability() throws Exception {
    Mutability mutability = Mutability.create("test");
    MutableList<Object> list = MutableList.copyOf(mutability, ImmutableList.of(1, 2, 3));
    SkylarkDict<Object, Object> dict = SkylarkDict.copyOf(mutability, ImmutableMap.of(1, 2, 3, 4));
    mutability.freeze();

    try {
      Iterator<?> it = list.iterator();
      it.next();
      it.remove();
      fail("expected exception");
    } catch (UnsupportedOperationException expected) {
    }
    try {
      Iterator<?> it = list.listIterator();
      it.next();
      it.remove();
      fail("expected exception");
    } catch (UnsupportedOperationException expected) {
    }
    try {
      Iterator<?> it = list.listIterator(1);
      it.next();
      it.remove();
      fail("expected exception");
    } catch (UnsupportedOperationException expected) {
    }
    try {
      List<Object> sublist = list.subList(1, 2);
      sublist.set(0, 4);
      fail("expected exception");
    } catch (UnsupportedOperationException expected) {
    }
    try {
      Iterator<Entry<Object, Object>> it = dict.entrySet().iterator();
      Entry<Object, Object> entry = it.next();
      entry.setValue(5);
      fail("expected exception");
    } catch (UnsupportedOperationException expected) {
    }
    try {
      Iterator<Object> it = dict.keySet().iterator();
      it.next();
      it.remove();
      fail("expected exception");
    } catch (UnsupportedOperationException expected) {
    }
    try {
      Iterator<Object> it = dict.values().iterator();
      it.next();
      it.remove();
      fail("expected exception");
    } catch (UnsupportedOperationException expected) {
    }
  }
}
