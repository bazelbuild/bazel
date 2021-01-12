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

package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkMutable}. */
@RunWith(JUnit4.class)
public final class StarlarkMutableTest {

  @Test
  public void testListViewsCheckMutability() throws Exception {
    Mutability mutability = Mutability.create("test");
    StarlarkList<Object> list =
        StarlarkList.copyOf(
            mutability, ImmutableList.of(StarlarkInt.of(1), StarlarkInt.of(2), StarlarkInt.of(3)));
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
    Dict<Object, Object> dict =
        Dict.copyOf(
            mutability,
            ImmutableMap.of(
                StarlarkInt.of(1), StarlarkInt.of(2), StarlarkInt.of(3), StarlarkInt.of(4)));
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

  @Test
  public void testDictBuilder() throws Exception {
    // put
    Dict<String, String> dict1 =
        Dict.<String, String>builder()
            .put("one", "1")
            .put("two", "2.0")
            .put("two", "2") // overrwrites previous entry
            .put("three", "3")
            .buildImmutable();
    assertThat(dict1.toString()).isEqualTo("{\"one\": \"1\", \"two\": \"2\", \"three\": \"3\"}");
    assertThrows(EvalException.class, dict1::clearEntries); // immutable

    // putAll
    Dict<String, String> dict2 =
        Dict.<String, String>builder()
            .putAll(dict1)
            .putAll(ImmutableMap.of("four", "4", "five", "5"))
            .buildImmutable();
    assertThat(dict2.toString())
        .isEqualTo(
            "{\"one\": \"1\", \"two\": \"2\", \"three\": \"3\", \"four\": \"4\", \"five\": \"5\"}");

    // builder reuse and mutability
    Dict.Builder<String, String> builder = Dict.<String, String>builder().putAll(dict1);
    Mutability mu = Mutability.create("test");
    Dict<String, String> dict3 = builder.build(mu);
    dict3.putEntry("four", "4");
    assertThat(dict3.toString())
        .isEqualTo("{\"one\": \"1\", \"two\": \"2\", \"three\": \"3\", \"four\": \"4\"}");
    mu.close();
    assertThrows(EvalException.class, dict1::clearEntries); // frozen
    builder.put("five", "5"); // keep building
    Dict<String, String> dict4 = builder.buildImmutable();
    assertThat(dict4.toString())
        .isEqualTo("{\"one\": \"1\", \"two\": \"2\", \"three\": \"3\", \"five\": \"5\"}");
    assertThat(dict3.toString())
        .isEqualTo(
            "{\"one\": \"1\", \"two\": \"2\", \"three\": \"3\", \"four\": \"4\"}"); // unchanged
  }
}
