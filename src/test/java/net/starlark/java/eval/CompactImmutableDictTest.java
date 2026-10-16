// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.testing.EqualsTester;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CompactImmutableDict}. */
@RunWith(JUnit4.class)
public final class CompactImmutableDictTest {

  @Test
  public void emptyDict() {
    CompactImmutableDict<String, Integer> dict = CompactImmutableDict.copyOf(ImmutableMap.of());
    performUniversalChecks(dict);

    assertThat(dict.isEmpty()).isTrue();
    assertThat(dict.size()).isEqualTo(0);
    assertThat(dict.get("a")).isNull();
    assertThat(dict.containsKey("a")).isFalse();
    assertThat(dict.containsValue(1)).isFalse();

    assertThat(dict.keySet()).isEmpty();
    assertThat(dict.values()).isEmpty();
    assertThat(dict.entrySet()).isEmpty();

    dict.forEach(
        (k, v) -> {
          throw new AssertionError("Should not be called");
        });
  }

  @Test
  public void singletonDict() {
    CompactImmutableDict<String, Integer> dict =
        CompactImmutableDict.copyOf(ImmutableMap.of("a", 1));
    performUniversalChecks(dict);

    assertThat(dict.isEmpty()).isFalse();
    assertThat(dict.size()).isEqualTo(1);
    assertThat(dict.get("a")).isEqualTo(1);
    assertThat(dict.get("b")).isNull();
    assertThat(dict.containsKey("a")).isTrue();
    assertThat(dict.containsKey("b")).isFalse();
    assertThat(dict.containsValue(1)).isTrue();
    assertThat(dict.containsValue(2)).isFalse();

    assertThat(dict.keySet()).containsExactly("a");
    assertThat(dict.values()).containsExactly(1);
    assertThat(dict.entrySet()).hasSize(1);

    Map<String, Integer> visited = new HashMap<>();
    dict.forEach(visited::put);
    assertThat(visited).containsExactly("a", 1);
  }

  @Test
  public void doubletonDict() {
    CompactImmutableDict<String, Integer> dict =
        CompactImmutableDict.copyOf(ImmutableMap.of("a", 1, "b", 2));
    performUniversalChecks(dict);

    assertThat(dict.isEmpty()).isFalse();
    assertThat(dict.size()).isEqualTo(2);
    assertThat(dict.get("a")).isEqualTo(1);
    assertThat(dict.get("b")).isEqualTo(2);
    assertThat(dict.get("c")).isNull();
    assertThat(dict.containsKey("a")).isTrue();
    assertThat(dict.containsKey("b")).isTrue();
    assertThat(dict.containsKey("c")).isFalse();
    assertThat(dict.containsValue(1)).isTrue();
    assertThat(dict.containsValue(2)).isTrue();
    assertThat(dict.containsValue(3)).isFalse();

    assertThat(dict.keySet()).containsExactly("a", "b").inOrder();
    assertThat(dict.values()).containsExactly(1, 2).inOrder();
    assertThat(dict.entrySet()).hasSize(2);

    Map<String, Integer> visited = new LinkedHashMap<>();
    dict.forEach(visited::put);
    assertThat(visited).containsExactly("a", 1, "b", 2).inOrder();
  }

  @Test
  public void linearDict() {
    Map<String, Integer> source = new LinkedHashMap<>();
    for (int i = 0; i < 6; i++) {
      source.put("k" + i, i);
    }
    CompactImmutableDict<String, Integer> dict = CompactImmutableDict.copyOf(source);
    performUniversalChecks(dict);

    assertThat(dict.isEmpty()).isFalse();
    assertThat(dict.size()).isEqualTo(6);
    assertThat(dict.get("k3")).isEqualTo(3);
    assertThat(dict.get("k9")).isNull();
    assertThat(dict.containsKey("k3")).isTrue();
    assertThat(dict.containsKey("k9")).isFalse();
    assertThat(dict.containsValue(3)).isTrue();
    assertThat(dict.containsValue(9)).isFalse();

    assertThat(dict.keySet()).containsExactly("k0", "k1", "k2", "k3", "k4", "k5").inOrder();
    assertThat(dict.values()).containsExactly(0, 1, 2, 3, 4, 5).inOrder();

    Map<String, Integer> visited = new LinkedHashMap<>();
    dict.forEach(visited::put);
    assertThat(visited).containsExactlyEntriesIn(source).inOrder();
  }

  @Test
  public void hashDict() {
    Map<String, Integer> source = new LinkedHashMap<>();
    for (int i = 0; i < 12; i++) {
      source.put("k" + i, i);
    }
    CompactImmutableDict<String, Integer> dict = CompactImmutableDict.copyOf(source);
    performUniversalChecks(dict);

    assertThat(dict.isEmpty()).isFalse();
    assertThat(dict.size()).isEqualTo(12);
    assertThat(dict.get("k7")).isEqualTo(7);
    assertThat(dict.get("k99")).isNull();
    assertThat(dict.containsKey("k7")).isTrue();
    assertThat(dict.containsKey("k99")).isFalse();
    assertThat(dict.containsValue(7)).isTrue();
    assertThat(dict.containsValue(99)).isFalse();

    assertThat(dict.keySet())
        .containsExactly("k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11")
        .inOrder();
    assertThat(dict.values()).containsExactly(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11).inOrder();

    Map<String, Integer> visited = new LinkedHashMap<>();
    dict.forEach(visited::put);
    assertThat(visited).containsExactlyEntriesIn(source).inOrder();
  }

  @Test
  public void iteratorNoSuchElementException() {
    CompactImmutableDict<String, Integer> dict =
        CompactImmutableDict.copyOf(ImmutableMap.of("a", 1, "b", 2, "c", 3));
    var it = dict.entrySet().iterator();

    assertThat(it.hasNext()).isTrue();
    it.next();
    assertThat(it.hasNext()).isTrue();
    it.next();
    assertThat(it.hasNext()).isTrue();
    it.next();
    assertThat(it.hasNext()).isFalse();

    assertThrows(NoSuchElementException.class, it::next);
  }

  @Test
  public void hashCollisions() {
    class BadHashKey {
      private final String val;

      BadHashKey(String val) {
        this.val = val;
      }

      @Override
      public int hashCode() {
        return 42; // Force every key to hash to the same primary slot!
      }

      @Override
      public boolean equals(Object obj) {
        return obj instanceof BadHashKey other && other.val.equals(val);
      }
    }

    Map<BadHashKey, Integer> source = new LinkedHashMap<>();
    for (int i = 0; i < 9; i++) {
      source.put(new BadHashKey("key" + i), i);
    }

    CompactImmutableDict<BadHashKey, Integer> dict = CompactImmutableDict.copyOf(source);

    // Verify all can be retrieved correctly (requires probing since every key collides!)
    for (int i = 0; i < 9; i++) {
      BadHashKey k = new BadHashKey("key" + i);
      assertThat(dict.get(k)).isEqualTo(i);
      assertThat(dict.containsKey(k)).isTrue();
    }

    // Verify absent key lookup works
    assertThat(dict.get(new BadHashKey("absent"))).isNull();
    assertThat(dict.containsKey(new BadHashKey("absent"))).isFalse();
  }

  @Test
  public void equalsAndHashCode() {
    new EqualsTester()
        // Empty
        .addEqualityGroup(createMapEqualityGroup(ImmutableMap.of()))
        // Singleton
        .addEqualityGroup(createMapEqualityGroup(ImmutableMap.of("a", "1")))
        .addEqualityGroup(createMapEqualityGroup(ImmutableMap.of("b", "2")))
        // Doubleton
        .addEqualityGroup(createMapEqualityGroup(ImmutableMap.of("a", "1", "b", "2")))
        .addEqualityGroup(createMapEqualityGroup(ImmutableMap.of("c", "3", "d", "4")))
        // Linear
        .addEqualityGroup(
            createMapEqualityGroup(
                ImmutableMap.of(
                    "a", "1",
                    "b", "2",
                    "c", "3")))
        .addEqualityGroup(
            createMapEqualityGroup(
                ImmutableMap.of(
                    "d", "4",
                    "e", "5",
                    "f", "6")))
        // Hash
        .addEqualityGroup(
            createMapEqualityGroup(
                ImmutableMap.of(
                    "a", "1",
                    "b", "2",
                    "c", "3",
                    "d", "4",
                    "e", "5",
                    "f", "6",
                    "g", "7",
                    "h", "8",
                    "i", "9")))
        .addEqualityGroup(
            createMapEqualityGroup(
                ImmutableMap.of(
                    "j", "10",
                    "k", "11",
                    "l", "12",
                    "m", "13",
                    "n", "14",
                    "o", "15",
                    "p", "16",
                    "q", "17",
                    "r", "18",
                    "s", "19")))
        .testEquals();
  }

  private static Object[] createMapEqualityGroup(ImmutableMap<String, String> m) {
    return new Object[] {
      // ImmutableMap
      m,
      // ImmutableMap with reverse order
      ImmutableMap.copyOf(m.entrySet().asList().reverse()),
      // MutableDict
      Dict.copyOf(Mutability.create(), m),
      // ImmutableMapBackedDict
      Dict.immutableCopyOf(m),
      // CompactImmutableDict
      CompactImmutableDict.copyOf(m)
    };
  }

  private static void performUniversalChecks(Dict<String, Integer> dict) {
    assertImmutable(dict);
    assertStarlarkMethodsRespectGivenMutablility(dict);
    assertNullSafeQueries(dict);
  }

  private static void assertImmutable(Dict<String, Integer> dict) {
    assertThrows(EvalException.class, () -> dict.putEntry("b", 2));
    assertThrows(EvalException.class, () -> dict.putEntries(ImmutableMap.of("b", 2)));
    assertThrows(EvalException.class, dict::clearEntries);
    assertThrows(EvalException.class, () -> dict.pop("a", null, null));
    assertThrows(EvalException.class, dict::popitem);
    assertThrows(EvalException.class, () -> dict.setdefault("b", 2));

    assertThrows(UnsupportedOperationException.class, () -> dict.put("b", 2));
    assertThrows(UnsupportedOperationException.class, () -> dict.putAll(ImmutableMap.of("b", 2)));
    assertThrows(UnsupportedOperationException.class, () -> dict.remove("a"));
    assertThrows(UnsupportedOperationException.class, dict::clear);
  }

  private static void assertStarlarkMethodsRespectGivenMutablility(Dict<?, ?> dict) {
    Mutability mu = Mutability.create();
    StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
    assertThat(dict.keys(thread).mutability()).isSameInstanceAs(mu);
    assertThat(dict.values0(thread).mutability()).isSameInstanceAs(mu);
    assertThat(dict.items(thread).mutability()).isSameInstanceAs(mu);
  }

  private static void assertNullSafeQueries(Dict<?, ?> dict) {
    assertThat(dict.containsKey(null)).isFalse();
    assertThat(dict.containsValue(null)).isFalse();
    assertThat(dict.get(null)).isNull();
    assertThat(dict.keySet().contains(null)).isFalse();
    assertThat(dict.values().contains(null)).isFalse();
    assertThat(dict.entrySet().contains(null)).isFalse();
    assertThat(dict.entrySet().contains(Maps.immutableEntry(null, null))).isFalse();
    if (!dict.isEmpty()) {
      var presentKeyNullValueEntry = Maps.immutableEntry(dict.iterator().next(), null);
      assertThat(dict.entrySet().contains(presentKeyNullValueEntry)).isFalse();
    }
  }
}
