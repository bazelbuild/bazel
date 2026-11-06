// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.LinkedHashMap;
import java.util.Map;
import net.starlark.java.eval.CompactImmutableDict;
import net.starlark.java.eval.Compactable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.Tuple;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link DictCodec}. */
@RunWith(TestParameterInjector.class)
public final class DictCodecTest {
  private final ObjectCodecs standardCodecs = new ObjectCodecs(AutoRegistry.get());

  /** Boundary sizes for each canonical {@link Dict} representation. */
  private enum SizeCase {
    EMPTY(0, "EmptyImmutableDict"),
    SINGLETON(1, "SingletonImmutableDict"),
    DOUBLETON(2, "DoubletonImmutableDict"),
    MIN_LINEAR(3, "LinearImmutableDict"),
    MAX_LINEAR(CompactImmutableDict.MAX_LINEAR_SIZE, "LinearImmutableDict"),
    MIN_HASH(CompactImmutableDict.MAX_LINEAR_SIZE + 1, "HashImmutableDict"),
    LARGE_HASH(4 * CompactImmutableDict.MAX_LINEAR_SIZE, "HashImmutableDict");

    private final int size;
    private final Class<?> expectedCanonicalClass;

    SizeCase(int size, String expectedClassName) {
      this.size = size;
      try {
        this.expectedCanonicalClass =
            Class.forName(CompactImmutableDict.class.getName() + "$" + expectedClassName);
      } catch (ClassNotFoundException e) {
        throw new LinkageError(expectedClassName, e);
      }
    }
  }

  @Test
  public void testCodec() throws Exception {
    Dict<Object, Object> aliasedInnerDict =
        Dict.immutableCopyOf(ImmutableMap.of("1", "2", "3", "4"));
    Dict.Builder<Object, Object> largeDictBuilder = Dict.builder();
    for (int i = CompactImmutableDict.MAX_LINEAR_SIZE + 1; i >= 1; i--) {
      largeDictBuilder.put(Tuple.of("k", StarlarkInt.of(i)), StarlarkInt.of(i * 10));
    }
    new SerializationTester(
            Dict.empty(),
            Dict.immutableCopyOf(ImmutableMap.of("only", "one")),
            Dict.immutableCopyOf(
                ImmutableMap.of("1", "2", "3", Dict.immutableCopyOf(ImmutableMap.of("4", "5")))),
            Dict.immutableCopyOf(ImmutableMap.of("10", aliasedInnerDict, "20", aliasedInnerDict)),
            largeDictBuilder.buildImmutable())
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(DictCodecTest::verifyDeserialization)
        .runTests();
  }

  @Test
  public void canonicalizesAllDictImplementationsAndSizes(@TestParameter SizeCase sizeCase)
      throws Exception {
    int size = sizeCase.size;
    Map<String, StarlarkInt> entries = createEntriesInReverseOrder(size);

    Mutability mu = Mutability.create("frozen");
    Dict<String, StarlarkInt> frozenMutable = Dict.copyOf(mu, entries);
    mu.freeze();
    Dict<String, StarlarkInt> immutableMapBacked =
        Dict.immutableCopyOf(ImmutableMap.copyOf(entries));
    Dict<String, StarlarkInt> compact =
        Dict.<String, StarlarkInt>builder().putAll(entries).buildImmutable();

    ByteString mutableBytes = standardCodecs.serialize(frozenMutable);
    ByteString mapBackedBytes = standardCodecs.serialize(immutableMapBacked);
    ByteString compactBytes = standardCodecs.serialize(compact);

    assertThat(mutableBytes).isEqualTo(compactBytes);
    assertThat(mapBackedBytes).isEqualTo(compactBytes);

    for (Dict<String, StarlarkInt> candidate :
        ImmutableList.of(frozenMutable, immutableMapBacked, compact)) {
      // The deserialized value round-trips one of the Dict<String, StarlarkInt> candidates.
      @SuppressWarnings("unchecked")
      var deserialized =
          (Dict<String, StarlarkInt>)
              standardCodecs.deserialize(standardCodecs.serialize(candidate));
      if (size == 0) {
        assertThat((Object) deserialized).isSameInstanceAs(Dict.empty());
      } else {
        assertThat((Object) deserialized).isInstanceOf(sizeCase.expectedCanonicalClass);
      }
      assertThat((Map<String, StarlarkInt>) deserialized)
          .containsExactlyEntriesIn(entries)
          .inOrder();
      for (Map.Entry<String, StarlarkInt> entry : entries.entrySet()) {
        assertThat(deserialized.containsKey(entry.getKey())).isTrue();
        assertThat(deserialized.get(entry.getKey())).isEqualTo(entry.getValue());
      }
      assertThat(deserialized.containsKey("absent_key")).isFalse();
      assertThat(deserialized.get("absent_key")).isNull();
    }
  }

  @Test
  public void optimizingMemoryLayout_doesNotResultInDifferentSerializedBytes(
      @TestParameter SizeCase sizeCase) throws Exception {
    int size = sizeCase.size;
    Map<String, StarlarkInt> entries = createEntriesInReverseOrder(size);
    Mutability mu = Mutability.create("opt_" + size);
    Dict<String, StarlarkInt> mutableDict = Dict.copyOf(mu, entries);
    mu.freeze();

    ByteString beforeBytes = standardCodecs.serialize(mutableDict);
    Object optimized = ((Compactable) mutableDict).unsafeOptimizeMemoryLayout();
    ByteString afterBytes = standardCodecs.serialize(mutableDict);
    ByteString optimizedBytes = standardCodecs.serialize(optimized);

    assertThat(beforeBytes).isEqualTo(afterBytes);
    assertThat(beforeBytes).isEqualTo(optimizedBytes);
  }

  @Test
  public void unfrozenDict_serializeThrows() {
    Mutability mu = Mutability.create("unfrozen");
    Dict<String, StarlarkInt> unfrozen = Dict.copyOf(mu, ImmutableMap.of("a", StarlarkInt.of(1)));
    SerializationException e =
        assertThrows(SerializationException.class, () -> standardCodecs.serialize(unfrozen));
    assertThat(e).hasMessageThat().contains("Trying to serialize unfrozen Dict");
  }

  @Test
  public void immutableKeyTrackingDict_serializeThrowsWithoutTrackingKeys() {
    Dict.ImmutableKeyTrackingDict<String, StarlarkInt> keyTracking =
        Dict.<String, StarlarkInt>builder()
            .put("a", StarlarkInt.of(1))
            .put("b", StarlarkInt.of(2))
            .buildImmutableWithKeyTracking();

    SerializationException e =
        assertThrows(SerializationException.class, () -> standardCodecs.serialize(keyTracking));
    assertThat(e).hasMessageThat().contains("Trying to serialize ImmutableKeyTrackingDict");
    assertThat(keyTracking.getAccessedKeys()).isEmpty();
  }

  @Test
  public void cyclicDicts_roundtripOk(
      // An empty dict cannot contain itself.
      @TestParameter({
            "SINGLETON",
            "DOUBLETON",
            "MIN_LINEAR",
            "MAX_LINEAR",
            "MIN_HASH",
            "LARGE_HASH"
          })
          SizeCase sizeCase)
      throws Exception {
    assertCyclicDictRoundtrip(sizeCase.size, sizeCase.expectedCanonicalClass);
  }

  private void assertCyclicDictRoundtrip(int totalSize, Class<?> expectedCanonicalClass)
      throws Exception {
    Mutability mu = Mutability.create("cycle_" + totalSize);
    Dict<Object, Object> cyclic = Dict.of(mu);
    ImmutableList.Builder<String> expectedKeys = ImmutableList.builder();
    for (int i = 0; i < totalSize - 1; i++) {
      String key = "key_" + i;
      cyclic.putEntry(key, StarlarkInt.of(i));
      expectedKeys.add(key);
    }
    cyclic.putEntry("self", cyclic);
    expectedKeys.add("self");
    mu.freeze();

    var deserialized =
        (Map<?, ?>) standardCodecs.deserializeMemoized(standardCodecs.serializeMemoized(cyclic));
    assertThat((Object) deserialized).isInstanceOf(expectedCanonicalClass);
    assertThat(deserialized).hasSize(totalSize);
    assertThat(deserialized.keySet()).containsExactlyElementsIn(expectedKeys.build()).inOrder();
    for (int i = 0; i < totalSize - 1; i++) {
      assertThat(deserialized.get("key_" + i)).isEqualTo(StarlarkInt.of(i));
    }
    assertThat(deserialized.get("self")).isSameInstanceAs(deserialized);
    assertThat(deserialized.containsKey("absent_key")).isFalse();
  }

  private static Map<String, StarlarkInt> createEntriesInReverseOrder(int size) {
    LinkedHashMap<String, StarlarkInt> entries = new LinkedHashMap<>();
    for (int i = size - 1; i >= 0; i--) {
      entries.put("key_" + i, StarlarkInt.of(i));
    }
    return entries;
  }

  // Check for order.
  private static void verifyDeserialization(
      Dict<Object, Object> subject, Map<Object, Object> deserialized) {
    assertThat(deserialized).isEqualTo(subject);
    assertThat(deserialized).containsExactlyEntriesIn(subject).inOrder();
  }
}
