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

package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.TruthJUnit.assume;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ImmutableSharedKeyMap}. */
@RunWith(TestParameterInjector.class)
public final class ImmutableSharedKeyMapTest {

  private enum CreationMode {
    BUILDER {
      @Override
      <K, V> ImmutableSharedKeyMap<K, V> createFrom(ImmutableMap<K, V> map) {
        var builder = ImmutableSharedKeyMap.<K, V>builder();
        map.forEach(builder::put);
        return builder.build();
      }
    },
    COPY_OF {
      @Override
      <K, V> ImmutableSharedKeyMap<K, V> createFrom(ImmutableMap<K, V> map) {
        return ImmutableSharedKeyMap.copyOf(map);
      }
    };

    abstract <K, V> ImmutableSharedKeyMap<K, V> createFrom(ImmutableMap<K, V> map);
  }

  @TestParameter private CreationMode creationMode;

  private <K, V> ImmutableSharedKeyMap<K, V> createFrom(ImmutableMap<K, V> map) {
    return creationMode.createFrom(map);
  }

  @Test
  public void testBasicFunctionality() {
    Object valueA = new Object();
    Object valueB = new Object();
    var immutableMap = ImmutableMap.of("a", valueA, "b", valueB);
    ImmutableSharedKeyMap<String, Object> map = createFrom(immutableMap);

    assertThat(map.get("a")).isSameInstanceAs(valueA);
    assertThat(map.get("b")).isSameInstanceAs(valueB);
    assertThat(map.get("c")).isNull();

    // Verify that we can find all items both by iteration and indexing
    ImmutableMap.Builder<String, Object> iterationCopy = ImmutableMap.builder();
    for (String key : map) {
      iterationCopy.put(key, map.get(key));
    }
    assertThat(iterationCopy.buildOrThrow()).isEqualTo(immutableMap);

    ImmutableMap.Builder<String, Object> arrayIterationCopy = ImmutableMap.builder();
    for (int i = 0; i < map.size(); ++i) {
      arrayIterationCopy.put(map.keyAt(i), map.valueAt(i));
    }
    assertThat(arrayIterationCopy.buildOrThrow()).isEqualTo(immutableMap);
  }

  @Test
  public void testEquality() {
    ImmutableSharedKeyMap<String, Object> emptyMap = createFrom(ImmutableMap.of());

    Object valueA = new Object();
    Object valueB = new Object();

    ImmutableSharedKeyMap<String, Object> map =
        createFrom(ImmutableMap.of("a", valueA, "b", valueB));

    // Two identically ordered maps are equal
    ImmutableSharedKeyMap<String, Object> exactCopy =
        createFrom(ImmutableMap.of("a", valueA, "b", valueB));

    // The map is order sensitive, so different insertion orders aren't equal
    ImmutableSharedKeyMap<String, Object> oppositeOrderMap =
        createFrom(ImmutableMap.of("b", valueB, "a", valueA));

    Object valueC = new Object();
    ImmutableSharedKeyMap<String, Object> biggerMap =
        createFrom(ImmutableMap.of("a", valueA, "b", valueB, "c", valueC));

    new EqualsTester()
        .addEqualityGroup(emptyMap)
        .addEqualityGroup(map, exactCopy)
        .addEqualityGroup(oppositeOrderMap)
        .addEqualityGroup(biggerMap)
        .testEquals();
  }

  @Test
  public void duplicateKeyPassedToBuilder_throws() {
    // This test only makes sense with the builder since copyOf takes a map which is duplicate-free.
    assume().that(creationMode).isEqualTo(CreationMode.BUILDER);

    Object valueA = new Object();
    Object valueB = new Object();
    Object valueC = new Object();
    ImmutableSharedKeyMap.Builder<String, Object> map =
        ImmutableSharedKeyMap.<String, Object>builder()
            .put("key", valueA)
            .put("key", valueB)
            .put("key", valueC);

    assertThrows(IllegalArgumentException.class, map::build);
  }

  private static final class SameHashCodeClass {
    @Override
    public int hashCode() {
      return 0;
    }
  }

  @Test
  public void twoKeysWithTheSameHashCode() {
    SameHashCodeClass keyA = new SameHashCodeClass();
    SameHashCodeClass keyB = new SameHashCodeClass();
    Object valueA = new Object();
    Object valueB = new Object();
    ImmutableSharedKeyMap<SameHashCodeClass, Object> map =
        createFrom(ImmutableMap.of(keyA, valueA, keyB, valueB));
    assertThat(map.get(keyA)).isSameInstanceAs(valueA);
    assertThat(map.get(keyB)).isSameInstanceAs(valueB);
  }
}
