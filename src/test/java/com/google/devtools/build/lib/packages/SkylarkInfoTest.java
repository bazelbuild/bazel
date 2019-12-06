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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.SkylarkInfo.Layout;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test class for {@link SkylarkInfo} and its subclasses. */
@RunWith(JUnit4.class)
public class SkylarkInfoTest {

  private static final Layout layoutF1F2 = new Layout(ImmutableList.of("f1", "f2"));
  private static final Layout invertedLayoutF2F1 = new Layout(ImmutableList.of("f2", "f1"));

  @Test
  public void layoutAccessors() {
    Layout layout = new Layout(ImmutableList.of("x", "y", "z"));
    assertThat(layout.size()).isEqualTo(3);
    assertThat(layout.getFieldIndex("a")).isNull();
    assertThat(layout.getFieldIndex("z")).isEqualTo(2);
    assertThat(layout.getFields()).containsExactly("x", "y", "z").inOrder();
    assertThat(
        layout.entrySet().stream()
            .map(Map.Entry::getKey)
            .collect(ImmutableList.toImmutableList()))
        .containsExactly("x", "y", "z").inOrder();
  }

  @Test
  public void layoutDisallowsDuplicates() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new Layout(ImmutableList.of("x", "y", "x")));
  }

  @Test
  public void layoutEquality() {
    new EqualsTester()
        .addEqualityGroup(
            new Layout(ImmutableList.of("a", "b", "c")),
            new Layout(ImmutableList.of("a", "b", "c")))
        .addEqualityGroup(
            new Layout(ImmutableList.of("x", "y", "z")))
        .addEqualityGroup(
            new Layout(ImmutableList.of("c", "b", "a")))
        .testEquals();
  }

  @Test
  public void nullLocationDefaultsToBuiltin() throws Exception {
    SkylarkInfo info = SkylarkInfo.createSchemaless(makeProvider(), ImmutableMap.of(), null);
    assertThat(info.getCreationLoc()).isEqualTo(Location.BUILTIN);
  }

  @Test
  public void givenLayoutTakesPrecedenceOverProviderLayout() throws Exception {
    SkylarkProvider provider =
        SkylarkProvider.createUnexportedSchemaful(ImmutableList.of("f1", "f2"), Location.BUILTIN);
    SkylarkInfo info =
        SkylarkInfo.createSchemaful(
            provider, invertedLayoutF2F1, new Object[]{5, 4}, Location.BUILTIN);
    assertThat(info.getLayout()).isEqualTo(invertedLayoutF2F1);  // not the one in the provider
  }

  @Test
  public void schemafulValuesMustMatchLayoutArity() throws Exception {
    SkylarkProvider provider = makeProvider();
    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                SkylarkInfo.createSchemaful(
                    provider, layoutF1F2, new Object[] {4}, Location.BUILTIN));
    assertThat(expected).hasMessageThat()
        .contains("Layout has length 2, but number of given values was 1");
  }

  @Test
  public void instancesOfUnexportedProvidersAreMutable() throws Exception {
    SkylarkProvider provider = makeProvider();
    SkylarkInfo mapInfo = makeSchemalessInfoWithF1F2Values(provider, 5, null);
    SkylarkInfo compactInfo = makeSchemafulInfoWithF1F2Values(provider, 5, null);
    assertThat(mapInfo.isImmutable()).isFalse();
    assertThat(compactInfo.isImmutable()).isFalse();
  }

  @Test
  public void instancesOfExportedProvidersMayBeImmutable() throws Exception {
    SkylarkProvider provider = makeExportedProvider();
    SkylarkInfo mapInfo = makeSchemalessInfoWithF1F2Values(provider, 5, null);
    SkylarkInfo compactInfo = makeSchemafulInfoWithF1F2Values(provider, 5, null);
    assertThat(mapInfo.isImmutable()).isTrue();
    assertThat(compactInfo.isImmutable()).isTrue();
  }

  @Test
  public void mutableIfContentsAreMutable() throws Exception {
    SkylarkProvider provider = makeExportedProvider();
    StarlarkValue v = new StarlarkValue() {};
    SkylarkInfo mapInfo = makeSchemalessInfoWithF1F2Values(provider, 5, v);
    SkylarkInfo compactInfo = makeSchemafulInfoWithF1F2Values(provider, 5, v);
    assertThat(mapInfo.isImmutable()).isFalse();
    assertThat(compactInfo.isImmutable()).isFalse();
  }

  @Test
  public void equality_DifferentProviders() throws Exception {
    SkylarkProvider provider1 = makeProvider();
    SkylarkProvider provider2 = makeProvider();
    new EqualsTester()
        .addEqualityGroup(
            makeSchemalessInfoWithF1F2Values(provider1, 4, 5),
            makeSchemafulInfoWithF1F2Values(provider1, 4, 5),
            makeInvertedSchemafulInfoWithF1F2Values(provider1, 4, 5))
        .addEqualityGroup(
            makeSchemalessInfoWithF1F2Values(provider2, 4, 5),
            makeSchemafulInfoWithF1F2Values(provider2, 4, 5),
            makeInvertedSchemafulInfoWithF1F2Values(provider2, 4, 5))
        .testEquals();
  }

  @Test
  public void equality_DifferentValues() throws Exception {
    SkylarkProvider provider = makeProvider();
    // These comparisons include the case where the physical array is {4, 5} on both instances but
    // they compare different due to different layouts.
    new EqualsTester()
        .addEqualityGroup(
            makeSchemalessInfoWithF1F2Values(provider, 4, 5),
            makeSchemafulInfoWithF1F2Values(provider, 4, 5),
            makeInvertedSchemafulInfoWithF1F2Values(provider, 4, 5))
        .addEqualityGroup(
            makeSchemalessInfoWithF1F2Values(provider, 5, 4),
            makeSchemafulInfoWithF1F2Values(provider, 5, 4),
            makeInvertedSchemafulInfoWithF1F2Values(provider, 5, 4))
        .addEqualityGroup(
            makeSchemalessInfoWithF1F2Values(provider, 4, null),
            makeSchemafulInfoWithF1F2Values(provider, 4, null),
            makeInvertedSchemafulInfoWithF1F2Values(provider, 4, null))
        .testEquals();
  }

  @Test
  public void concatWithDifferentProvidersFails() throws Exception {
    SkylarkProvider provider1 = makeProvider();
    SkylarkProvider provider2 = makeProvider();
    SkylarkInfo info1 = makeSchemalessInfoWithF1F2Values(provider1, 4, 5);
    SkylarkInfo info2 = makeSchemalessInfoWithF1F2Values(provider2, 4, 5);
    EvalException expected =
        assertThrows(
            EvalException.class, () -> info1.getConcatter().concat(info1, info2, Location.BUILTIN));
    assertThat(expected).hasMessageThat()
        .contains("Cannot use '+' operator on instances of different providers");
  }

  @Test
  public void concatWithOverlappingFieldsFails() throws Exception {
    SkylarkProvider provider1 = makeProvider();
    SkylarkInfo info1 = makeSchemalessInfoWithF1F2Values(provider1, 4, 5);
    SkylarkInfo info2 = makeSchemalessInfoWithF1F2Values(provider1, 4, null);
    EvalException expected =
        assertThrows(
            EvalException.class, () -> info1.getConcatter().concat(info1, info2, Location.BUILTIN));
    assertThat(expected).hasMessageThat()
        .contains("Cannot use '+' operator on provider instances with overlapping field(s): f1");
  }

  @Test
  public void compactConcatReturnsCompact() throws Exception {
    SkylarkProvider provider = makeProvider();
    SkylarkInfo info1 = makeSchemafulInfoWithF1F2Values(provider, 4, null);
    SkylarkInfo info2 = makeSchemafulInfoWithF1F2Values(provider, null, 5);
    SkylarkInfo result = (SkylarkInfo) info1.getConcatter().concat(info1, info2, Location.BUILTIN);
    assertThat(result.isCompact()).isTrue();
    assertThat(result.getFieldNames()).containsExactly("f1", "f2");
    assertThat(result.getValue("f1")).isEqualTo(4);
    assertThat(result.getValue("f2")).isEqualTo(5);
  }

  @Test
  public void compactConcatWithDifferentLayoutsReturnsMap() throws Exception {
    SkylarkProvider provider = makeProvider();
    SkylarkInfo info1 = makeSchemafulInfoWithF1F2Values(provider, 4, null);
    SkylarkInfo info2 = makeInvertedSchemafulInfoWithF1F2Values(provider, null, 5);
    SkylarkInfo result = (SkylarkInfo) info1.getConcatter().concat(info1, info2, Location.BUILTIN);
    assertThat(result.isCompact()).isFalse();
    assertThat(result.getFieldNames()).containsExactly("f1", "f2");
    assertThat(result.getValue("f1")).isEqualTo(4);
    assertThat(result.getValue("f2")).isEqualTo(5);
  }

  @Test
  public void allOtherConcatReturnsMap() throws Exception {
    SkylarkProvider provider = makeProvider();
    SkylarkInfo info1 = makeSchemalessInfoWithF1F2Values(provider, 4, null);
    SkylarkInfo info2 = makeSchemafulInfoWithF1F2Values(provider, null, 5);
    SkylarkInfo result = (SkylarkInfo) info1.getConcatter().concat(info1, info2, Location.BUILTIN);
    assertThat(result.isCompact()).isFalse();
    assertThat(result.getFieldNames()).containsExactly("f1", "f2");
    assertThat(result.getValue("f1")).isEqualTo(4);
    assertThat(result.getValue("f2")).isEqualTo(5);
  }

  /** Creates an unexported schemaless provider type with builtin location. */
  private static SkylarkProvider makeProvider() {
    return SkylarkProvider.createUnexportedSchemaless(Location.BUILTIN);
  }

  /** Creates an exported schemaless provider type with builtin location. */
  private static SkylarkProvider makeExportedProvider() {
    SkylarkProvider.SkylarkKey key = new SkylarkProvider.SkylarkKey(
        Label.parseAbsoluteUnchecked("//package:target"), "provider");
    return SkylarkProvider.createExportedSchemaless(key, Location.BUILTIN);
  }

  /**
   * Creates a schemaless instance of a provider with the given values for fields f1 and f2. Either
   * field value may be null, in which case it is omitted.
   */
  private static SkylarkInfo makeSchemalessInfoWithF1F2Values(
      SkylarkProvider provider, @Nullable Object v1, @Nullable Object v2) {
    ImmutableMap.Builder<String, Object> values = ImmutableMap.builder();
    if (v1 != null) {
      values.put("f1", v1);
    }
    if (v2 != null) {
      values.put("f2", v2);
    }
    return SkylarkInfo.createSchemaless(provider, values.build(), Location.BUILTIN);
  }

  /**
   * Creates a schemaful instance of a provider with the given values for fields f1 and f2. Either
   * field value may be null, in which case it is omitted.
   */
  private static SkylarkInfo makeSchemafulInfoWithF1F2Values(
      SkylarkProvider provider, @Nullable Object v1, @Nullable Object v2) {
    return SkylarkInfo.createSchemaful(
        provider, layoutF1F2, new Object[]{v1, v2}, Location.BUILTIN);
  }

  /**
   * Same as {@link #makeSchemafulInfoWithF1F2Values}, except the layout in the resulting
   * CompactSkylarkInfo is reversed.
   */
  private static SkylarkInfo makeInvertedSchemafulInfoWithF1F2Values(
      SkylarkProvider provider, @Nullable Object v1, @Nullable Object v2) {
    return SkylarkInfo.createSchemaful(
        provider, invertedLayoutF2F1, new Object[]{v2, v1}, Location.BUILTIN);
  }
}
