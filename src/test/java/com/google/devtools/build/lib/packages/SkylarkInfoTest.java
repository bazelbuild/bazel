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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test class for {@link SkylarkInfo} and its subclasses. */
@RunWith(JUnit4.class)
public class SkylarkInfoTest {

  @Test
  public void nullLocationDefaultsToBuiltin() throws Exception {
    SkylarkInfo info = SkylarkInfo.create(makeProvider(), ImmutableMap.of(), null);
    assertThat(info.getCreationLoc()).isEqualTo(Location.BUILTIN);
  }

  @Test
  public void instancesOfUnexportedProvidersAreMutable() throws Exception {
    SkylarkProvider provider = makeProvider();
    SkylarkInfo info = makeInfoWithF1F2Values(provider, 5, null);
    assertThat(info.isImmutable()).isFalse();
  }

  @Test
  public void instancesOfExportedProvidersMayBeImmutable() throws Exception {
    SkylarkProvider provider = makeExportedProvider();
    SkylarkInfo info = makeInfoWithF1F2Values(provider, 5, null);
    assertThat(info.isImmutable()).isTrue();
  }

  @Test
  public void mutableIfContentsAreMutable() throws Exception {
    SkylarkProvider provider = makeExportedProvider();
    StarlarkValue v = new StarlarkValue() {};
    SkylarkInfo info = makeInfoWithF1F2Values(provider, 5, v);
    assertThat(info.isImmutable()).isFalse();
  }

  @Test
  public void equivalence() throws Exception {
    SkylarkProvider provider1 = makeProvider();
    SkylarkProvider provider2 = makeProvider();
    // equal providers and fields
    assertThat(makeInfoWithF1F2Values(provider1, 4, 5))
        .isEqualTo(makeInfoWithF1F2Values(provider1, 4, 5));
    // different providers => unequal
    assertThat(makeInfoWithF1F2Values(provider1, 4, 5))
        .isNotEqualTo(makeInfoWithF1F2Values(provider2, 4, 5));
    // different fields => unequal
    assertThat(makeInfoWithF1F2Values(provider1, 4, 5))
        .isNotEqualTo(makeInfoWithF1F2Values(provider1, 4, 6));
    // different sets of fields => unequal
    assertThat(makeInfoWithF1F2Values(provider1, 4, 5))
        .isNotEqualTo(makeInfoWithF1F2Values(provider1, 4, null));
  }

  @Test
  public void concatWithDifferentProvidersFails() throws Exception {
    SkylarkProvider provider1 = makeProvider();
    SkylarkProvider provider2 = makeProvider();
    SkylarkInfo info1 = makeInfoWithF1F2Values(provider1, 4, 5);
    SkylarkInfo info2 = makeInfoWithF1F2Values(provider2, 4, 5);
    EvalException expected =
        assertThrows(
            EvalException.class, () -> info1.getConcatter().concat(info1, info2, Location.BUILTIN));
    assertThat(expected).hasMessageThat()
        .contains("Cannot use '+' operator on instances of different providers");
  }

  @Test
  public void concatWithOverlappingFieldsFails() throws Exception {
    SkylarkProvider provider1 = makeProvider();
    SkylarkInfo info1 = makeInfoWithF1F2Values(provider1, 4, 5);
    SkylarkInfo info2 = makeInfoWithF1F2Values(provider1, 4, null);
    EvalException expected =
        assertThrows(
            EvalException.class, () -> info1.getConcatter().concat(info1, info2, Location.BUILTIN));
    assertThat(expected)
        .hasMessageThat()
        .contains("cannot add struct instances with common field 'f1'");
  }

  @Test
  public void concatWithSameFields() throws Exception {
    SkylarkProvider provider = makeProvider();
    SkylarkInfo info1 = makeInfoWithF1F2Values(provider, 4, null);
    SkylarkInfo info2 = makeInfoWithF1F2Values(provider, null, 5);
    SkylarkInfo result = (SkylarkInfo) info1.getConcatter().concat(info1, info2, Location.BUILTIN);
    assertThat(result.getFieldNames()).containsExactly("f1", "f2");
    assertThat(result.getValue("f1")).isEqualTo(4);
    assertThat(result.getValue("f2")).isEqualTo(5);
  }

  @Test
  public void concatWithDifferentFields() throws Exception {
    SkylarkProvider provider = makeProvider();
    SkylarkInfo info1 = makeInfoWithF1F2Values(provider, 4, null);
    SkylarkInfo info2 = makeInfoWithF1F2Values(provider, null, 5);
    SkylarkInfo result = (SkylarkInfo) info1.getConcatter().concat(info1, info2, Location.BUILTIN);
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
   * Creates an instance of a provider with the given values for fields f1 and f2. Either field
   * value may be null, in which case it is omitted.
   */
  private static SkylarkInfo makeInfoWithF1F2Values(
      SkylarkProvider provider, @Nullable Object v1, @Nullable Object v2) {
    ImmutableMap.Builder<String, Object> values = ImmutableMap.builder();
    if (v1 != null) {
      values.put("f1", v1);
    }
    if (v2 != null) {
      values.put("f2", v2);
    }
    return SkylarkInfo.create(provider, values.build(), Location.BUILTIN);
  }

}
