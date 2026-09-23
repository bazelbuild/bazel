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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.TokenKind;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test class for {@link StarlarkInfo} and its subclasses. */
@RunWith(JUnit4.class)
public class StarlarkInfoTest {

  @Test
  public void instancesOfUnexportedProvidersAreMutable() {
    StarlarkProvider provider = makeProvider();
    StarlarkInfo info = makeInfoWithF1F2Values(provider, StarlarkInt.of(5), null);
    assertThat(info.isImmutable()).isFalse();
  }

  @Test
  public void instancesOfExportedProvidersMayBeImmutable() {
    StarlarkProvider provider = makeExportedProvider();
    StarlarkInfo info = makeInfoWithF1F2Values(provider, StarlarkInt.of(5), null);
    assertThat(info.isImmutable()).isTrue();
  }

  @Test
  public void mutableIfContentsAreMutable() {
    StarlarkProvider provider = makeExportedProvider();
    StarlarkValue v = new StarlarkValue() {};
    StarlarkInfo info = makeInfoWithF1F2Values(provider, StarlarkInt.of(5), v);
    assertThat(info.isImmutable()).isFalse();
  }

  @Test
  public void equivalence() throws Exception {
    StarlarkProvider provider1 = makeProvider();
    StarlarkProvider provider2 = makeProvider();
    // equal providers and fields
    assertThat(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(5)))
        .isEqualTo(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(5)));
    // different providers => unequal
    assertThat(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(5)))
        .isNotEqualTo(makeInfoWithF1F2Values(provider2, StarlarkInt.of(4), StarlarkInt.of(5)));
    // different fields => unequal
    assertThat(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(5)))
        .isNotEqualTo(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(6)));
    // different sets of fields => unequal
    assertThat(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(5)))
        .isNotEqualTo(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), null));
    // different field names with the same values => unequal
    assertThat(makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), null))
        .isNotEqualTo(makeInfoWithF1F2Values(provider1, null, StarlarkInt.of(4)));
  }

  @Test
  public void concatWithDifferentProvidersFails() {
    StarlarkProvider provider1 = makeProvider();
    StarlarkProvider provider2 = makeProvider();
    StarlarkInfo info1 = makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(5));
    StarlarkInfo info2 = makeInfoWithF1F2Values(provider2, StarlarkInt.of(4), StarlarkInt.of(5));
    EvalException expected =
        assertThrows(EvalException.class, () -> info1.binaryOp(TokenKind.PLUS, info2, true));
    assertThat(expected)
        .hasMessageThat()
        .contains("Cannot use '+' operator on instances of different providers");
  }

  @Test
  public void concatWithOverlappingFieldsFails() {
    StarlarkProvider provider1 = makeProvider();
    StarlarkInfo info1 = makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), StarlarkInt.of(5));
    StarlarkInfo info2 = makeInfoWithF1F2Values(provider1, StarlarkInt.of(4), null);
    EvalException expected =
        assertThrows(EvalException.class, () -> info1.binaryOp(TokenKind.PLUS, info2, true));
    assertThat(expected)
        .hasMessageThat()
        .contains("cannot add struct instances with common field 'f1'");
  }

  @Test
  public void concatWithSameFields() throws Exception {
    StarlarkProvider provider = makeProvider();
    StarlarkInfo info1 = makeInfoWithF1F2Values(provider, StarlarkInt.of(4), null);
    StarlarkInfo info2 = makeInfoWithF1F2Values(provider, null, StarlarkInt.of(5));
    StarlarkInfo result = (StarlarkInfo) info1.binaryOp(TokenKind.PLUS, info2, true);
    assertThat(result.getFieldNames()).containsExactly("f1", "f2");
    assertThat(result.getValue("f1")).isEqualTo(StarlarkInt.of(4));
    assertThat(result.getValue("f2")).isEqualTo(StarlarkInt.of(5));
  }

  @Test
  public void concatWithDifferentFields() throws Exception {
    StarlarkProvider provider = makeProvider();
    StarlarkInfo info1 = makeInfoWithF1F2Values(provider, StarlarkInt.of(4), null);
    StarlarkInfo info2 = makeInfoWithF1F2Values(provider, null, StarlarkInt.of(5));
    StarlarkInfo result = (StarlarkInfo) info1.binaryOp(TokenKind.PLUS, info2, true);
    assertThat(result.getFieldNames()).containsExactly("f1", "f2");
    assertThat(result.getValue("f1")).isEqualTo(StarlarkInt.of(4));
    assertThat(result.getValue("f2")).isEqualTo(StarlarkInt.of(5));
  }

  /** Creates an unexported schemaless provider type with builtin location. */
  private static StarlarkProvider makeProvider() {
    return StarlarkProvider.builder(Location.BUILTIN)
        .buildWithIdentityToken(SymbolGenerator.createTransient().generate());
  }

  /** Creates an exported schemaless provider type with builtin location. */
  private static StarlarkProvider makeExportedProvider() {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(
            keyForBuild(Label.parseCanonicalUnchecked("//package:target")), "provider");
    return StarlarkProvider.builder(Location.BUILTIN).buildExported(key);
  }

  /**
   * Creates an instance of a provider with the given values for fields f1 and f2. Either field
   * value may be null, in which case it is omitted.
   */
  private static StarlarkInfo makeInfoWithF1F2Values(
      StarlarkProvider provider, @Nullable Object v1, @Nullable Object v2) {
    ImmutableMap.Builder<String, Object> values = ImmutableMap.builder();
    if (v1 != null) {
      values.put("f1", v1);
    }
    if (v2 != null) {
      values.put("f2", v2);
    }
    return StarlarkInfo.create(provider, values.buildOrThrow());
  }

  @Test
  public void compactionReusesSchemaStorageAndSharesSchemas() throws Exception {
    StarlarkProvider provider = makeExportedProvider();
    for (int size : new int[] {0, 1, 2, 3, 4, 5, 6, 16, 17, 64}) {
      Map<String, Object> values = new LinkedHashMap<>();
      for (int i = size - 1; i >= 0; i--) {
        values.put("field" + i, StarlarkInt.of(i));
      }
      var names = new ArrayList<>(values.keySet());
      Collections.sort(names);
      StarlarkInfo original = StarlarkInfo.create(provider, values);
      StarlarkInfo peer = StarlarkInfo.create(provider, values);
      Map<StarlarkInfo, String> map = new HashMap<>();
      map.put(original, "value");
      int hash = original.hashCode();
      StarlarkInfo compact = original.unsafeOptimizeMemoryLayout();
      for (StarlarkInfo info : new StarlarkInfo[] {original, compact}) {
        assertThat(info.getFieldNames()).containsExactlyElementsIn(names).inOrder();
        for (String name : names) {
          assertThat(info.getValue(name)).isEqualTo(values.get(name));
        }
        assertThat(info.getValue("missing")).isNull();
      }
      assertThat(original).isInstanceOf(StarlarkInfoNoSchema.class);
      assertThat(compact).isInstanceOf(StarlarkInfoWithSchema.class);
      assertThat(compact.getClass().getSimpleName()).isEqualTo("Schema" + (size <= 5 ? size : "N"));
      assertThat(compact.getProvider()).isSameInstanceAs(provider);
      assertThat(compact).isEqualTo(original);
      assertThat(original).isEqualTo(compact);
      assertThat(compact.hashCode()).isEqualTo(hash);
      assertThat(original.hashCode()).isEqualTo(hash);
      assertThat(map.get(compact)).isEqualTo("value");
      StarlarkInfo compactPeer = peer.unsafeOptimizeMemoryLayout();
      assertThat(schemaOf(compact)).isSameInstanceAs(schemaOf(compactPeer));
      assertThat(compact).isEqualTo(compactPeer);
      assertThat(compact.unsafeOptimizeMemoryLayout()).isSameInstanceAs(compact);
    }
  }

  @Test
  public void inferredSchemasPreserveDistinctEqualProviders() {
    StarlarkProvider firstProvider = makeExportedProvider();
    StarlarkProvider secondProvider = makeExportedProvider();
    assertThat(firstProvider).isEqualTo(secondProvider);
    StarlarkInfo first =
        StarlarkInfo.create(firstProvider, ImmutableMap.of()).unsafeOptimizeMemoryLayout();
    StarlarkInfo second =
        StarlarkInfo.create(secondProvider, ImmutableMap.of()).unsafeOptimizeMemoryLayout();
    assertThat(first.getProvider()).isSameInstanceAs(firstProvider);
    assertThat(second.getProvider()).isSameInstanceAs(secondProvider);
    assertThat(first).isEqualTo(second);
    assertThat(first.hashCode()).isEqualTo(second.hashCode());
  }

  @Test
  public void inferredAndDeclaredSchemasRemainDistinct() {
    StarlarkProvider declaredProvider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(List.of("f1"))
            .buildExported(
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonicalUnchecked("//package:target")), "provider"));
    StarlarkInfo declared =
        StarlarkInfoWithSchema.create(declaredProvider, new Object[] {StarlarkInt.of(1)});
    StarlarkInfo original = makeInfoWithF1F2Values(makeExportedProvider(), StarlarkInt.of(1), null);
    StarlarkInfo compact = original.unsafeOptimizeMemoryLayout();
    assertThat(declared.getProvider()).isEqualTo(compact.getProvider());
    assertThat(declared).isNotEqualTo(original);
    assertThat(original).isNotEqualTo(declared);
    assertThat(declared).isNotEqualTo(compact);
    assertThat(compact).isNotEqualTo(declared);
  }

  @Test
  public void inferredSchemasRemainSharedAfterProviderExport() throws Exception {
    StarlarkProvider provider = makeProvider();
    StarlarkInfo original = makeInfoWithF1F2Values(provider, StarlarkInt.of(1), null);
    StarlarkInfo compact = original.unsafeOptimizeMemoryLayout();
    assertThat(compact.isImmutable()).isFalse();
    provider.export(
        event -> {}, Label.parseCanonicalUnchecked("//foo:bar.bzl"), "foo", Location.BUILTIN);
    StarlarkInfo peer =
        makeInfoWithF1F2Values(provider, StarlarkInt.of(2), null).unsafeOptimizeMemoryLayout();
    assertThat(compact.isImmutable()).isTrue();
    assertThat(compact.getProvider()).isSameInstanceAs(provider);
    assertThat(schemaOf(compact)).isSameInstanceAs(schemaOf(peer));
    assertThat(compact).isEqualTo(original);
    assertThat(compact.hashCode()).isEqualTo(original.hashCode());
  }

  @Test
  public void compactedStructsConcatenateWithUncompactedStructs() throws Exception {
    for (boolean compactLeft : new boolean[] {false, true}) {
      for (boolean compactRight : new boolean[] {false, true}) {
        StarlarkProvider provider = makeExportedProvider();
        StarlarkInfo left = makeInfoWithF1F2Values(provider, StarlarkInt.of(1), null);
        StarlarkInfo right = makeInfoWithF1F2Values(provider, null, StarlarkInt.of(1));
        if (compactLeft) {
          left = left.unsafeOptimizeMemoryLayout();
        }
        if (compactRight) {
          right = right.unsafeOptimizeMemoryLayout();
        }
        assertThat(left).isNotEqualTo(right);
        assertThat(right).isNotEqualTo(left);
        StarlarkInfo result = (StarlarkInfo) left.binaryOp(TokenKind.PLUS, right, true);
        StarlarkInfo reversed = (StarlarkInfo) right.binaryOp(TokenKind.PLUS, left, false);
        StarlarkInfo expected =
            makeInfoWithF1F2Values(provider, StarlarkInt.of(1), StarlarkInt.of(1));
        assertThat(result).isEqualTo(expected);
        assertThat(reversed).isEqualTo(expected);
        assertThat(result.hashCode()).isEqualTo(expected.hashCode());
        StarlarkInfo compactResult = result.unsafeOptimizeMemoryLayout();
        assertThat(compactResult).isEqualTo(expected);
        assertThat(compactResult.hashCode()).isEqualTo(expected.hashCode());
        StarlarkInfo overlapping = left;
        assertThrows(
            EvalException.class, () -> compactResult.binaryOp(TokenKind.PLUS, overlapping, true));
      }
    }
  }

  @Test
  public void compactionPreservesCustomMessagesAndNestedStructs() throws Exception {
    StarlarkProvider provider = makeExportedProvider();
    StarlarkInfo nested = makeInfoWithF1F2Values(provider, StarlarkInt.of(1), null);
    StarlarkInfo peer =
        makeInfoWithF1F2Values(provider, StarlarkInt.of(2), null).unsafeOptimizeMemoryLayout();
    StarlarkInfo outer =
        StructProvider.STRUCT.create(ImmutableMap.of("nested", nested), "missing %s");
    String error = outer.getErrorMessageForUnknownField("absent");
    StarlarkInfo compact = outer.unsafeOptimizeMemoryLayout();
    assertThat(compact.getErrorMessageForUnknownField("absent")).isEqualTo(error);
    assertThat(compact.getValue("nested")).isEqualTo(nested);
    assertThat(schemaOf((StarlarkInfo) compact.getValue("nested")))
        .isSameInstanceAs(schemaOf(peer));
    assertThat(compact).isEqualTo(outer);
    assertThat(compact.hashCode()).isEqualTo(outer.hashCode());
    StarlarkInfo emptyA =
        StructProvider.STRUCT.create(ImmutableMap.of(), "first %s").unsafeOptimizeMemoryLayout();
    StarlarkInfo emptyB =
        StructProvider.STRUCT.create(ImmutableMap.of(), "second %s").unsafeOptimizeMemoryLayout();
    assertThat(emptyA).isEqualTo(emptyB);
    assertThat(emptyA.hashCode()).isEqualTo(emptyB.hashCode());
    assertThat(emptyA.unsafeOptimizeMemoryLayout().getErrorMessageForUnknownField("x"))
        .startsWith("first x");
    assertThat(emptyB.unsafeOptimizeMemoryLayout().getErrorMessageForUnknownField("x"))
        .startsWith("second x");
  }

  @Test
  public void compactionPreservesDepsets() {
    for (Depset depset :
        new Depset[] {
          Depset.of(String.class, NestedSetBuilder.emptySet(Order.STABLE_ORDER)),
          Depset.of(String.class, NestedSetBuilder.<String>stableOrder().add("value").build())
        }) {
      StarlarkInfo original =
          StarlarkInfo.create(StructProvider.STRUCT, ImmutableMap.of("deps", depset));
      StarlarkInfo compact = original.unsafeOptimizeMemoryLayout();
      assertThat(compact.getValue("deps")).isSameInstanceAs(depset);
      assertThat(compact).isEqualTo(original);
      assertThat(original).isEqualTo(compact);
      assertThat(compact.hashCode()).isEqualTo(original.hashCode());
    }
  }

  private static Object schemaOf(StarlarkInfo info) throws Exception {
    var field = StarlarkInfoWithSchema.class.getDeclaredField("schema");
    field.setAccessible(true);
    return field.get(info);
  }

  // Tests sortPairs using arrays of various lengths from Fibonacci sequence.
  @Test
  public void testSortPairs() {
    boolean ok = true;
    Random rand = new Random(0);

    // (a, b) is the Fibonacci generator. We use a as the array length.
    for (int a = 0, b = 1; a < 1000; ) {
      // generate random array of a pairs.
      Object[] array = new Object[2 * a];
      for (int i = 0; i < a; i++) {
        int r = rand.nextInt(1000000);
        array[i] = String.format("key%06d", r);
        array[a + i] = r;
      }

      // Sort keys and values using reference implementation.
      @SuppressWarnings("unchecked")
      List<String> origKeys =
          (List<String>) (List<?>) new ArrayList<>(Arrays.asList(array).subList(0, a));
      Collections.sort(origKeys);
      @SuppressWarnings("unchecked")
      List<Integer> origValues =
          (List<Integer>) (List<?>) new ArrayList<>(Arrays.asList(array).subList(a, 2 * a));
      Collections.sort(origValues);

      // Sort using sortPairs.
      if (a > 0) {
        StarlarkInfoNoSchema.sortPairs(array, 0, a - 1);
      }

      // Assert sorted keys match reference implementation.
      List<?> keys = Arrays.asList(array).subList(0, a);
      if (!keys.equals(origKeys)) {
        System.err.printf("a=%d: keys not in order: got %s, want %s\n", a, keys, origKeys);
        ok = false;
      }

      // Assert sorted values match reference implementation.
      List<?> values = Arrays.asList(array).subList(a, 2 * a);
      if (!values.equals(origValues)) {
        System.err.printf("a=%d: values not in order: got %s, want %s\n", a, values, origValues);
        ok = false;
      }

      // next Fibonacci number
      int c = a + b;
      a = b;
      b = c;
    }
    if (!ok) {
      throw new AssertionError("failed");
    }
  }
}
