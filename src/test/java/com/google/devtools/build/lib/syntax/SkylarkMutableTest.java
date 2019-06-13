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

import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkMutable}. */
@RunWith(JUnit4.class)
public final class SkylarkMutableTest {

  @Test
  public void testListViewsCheckMutability() throws Exception {
    Mutability mutability = Mutability.create("test");
    MutableList<Object> list = MutableList.copyOf(mutability, ImmutableList.of(1, 2, 3));
    mutability.freeze();

    {
      Iterator<?> it = list.iterator();
      it.next();
      assertThrows(
          UnsupportedOperationException.class,
          () -> it.remove());
    }
    {
      Iterator<?> it = list.listIterator();
      it.next();
      assertThrows(
          UnsupportedOperationException.class,
          () -> it.remove());
    }
    {
      Iterator<?> it = list.listIterator(1);
      it.next();
      assertThrows(
          UnsupportedOperationException.class,
          () -> it.remove());
    }
    {
      List<Object> sublist = list.subList(1, 2);
      assertThrows(
          UnsupportedOperationException.class,
          () -> sublist.set(0, 4));
    }
  }

  @Test
  public void testDictViewsCheckMutability() throws Exception {
    Mutability mutability = Mutability.create("test");
    SkylarkDict<Object, Object> dict = SkylarkDict.copyOf(mutability, ImmutableMap.of(1, 2, 3, 4));
    mutability.freeze();

    {
      Iterator<Map.Entry<Object, Object>> it = dict.entrySet().iterator();
      Map.Entry<Object, Object> entry = it.next();
      assertThrows(
          UnsupportedOperationException.class,
          () -> entry.setValue(5));
    }
    {
      Iterator<Object> it = dict.keySet().iterator();
      it.next();
      assertThrows(
          UnsupportedOperationException.class,
          () -> it.remove());
    }
    {
      Iterator<Object> it = dict.values().iterator();
      it.next();
      assertThrows(
          UnsupportedOperationException.class,
          () -> it.remove());
    }
  }
}
