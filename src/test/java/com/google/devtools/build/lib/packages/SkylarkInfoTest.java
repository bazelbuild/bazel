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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.SkylarkInfo.CompactSkylarkInfo;
import com.google.devtools.build.lib.packages.SkylarkInfo.MapBackedSkylarkInfo;
import com.google.devtools.build.lib.syntax.Concatable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test class for {@link SkylarkInfo} and its subclasses. */
@RunWith(JUnit4.class)
public class SkylarkInfoTest {

  @Test
  public void sameProviderDifferentLayoutConcatenation() throws Exception {
    SkylarkProvider provider =
        SkylarkProvider.createUnexportedSchemaful(ImmutableList.of("f1", "f2"), Location.BUILTIN);
    ImmutableMap<String, Integer> layout1 = ImmutableMap.of("f1", 0, "f2", 1);
    ImmutableMap<String, Integer> layout2 = ImmutableMap.of("f1", 1, "f2", 0);
    CompactSkylarkInfo info1 =
        new CompactSkylarkInfo(provider, layout1, new Object[] {5, null}, Location.BUILTIN);
    CompactSkylarkInfo info2 =
        new CompactSkylarkInfo(provider, layout2, new Object[] {4, null}, Location.BUILTIN);
    SkylarkInfo result = (SkylarkInfo) info1.getConcatter().concat(info1, info2, Location.BUILTIN);
    assertThat(result).isInstanceOf(MapBackedSkylarkInfo.class);
    assertThat(result.getValue("f1")).isEqualTo(5);
    assertThat(result.getValue("f2")).isEqualTo(4);
  }

  @Test
  public void immutabilityPredicate() throws Exception {
    SkylarkProvider provider =
        SkylarkProvider.createUnexportedSchemaful(ImmutableList.of("f1", "f2"), Location.BUILTIN);
    ImmutableMap<String, Integer> layout = ImmutableMap.of("f1", 0, "f2", 1);
    SkylarkInfo compactInfo =
        new CompactSkylarkInfo(provider, layout, new Object[] {5, null}, Location.BUILTIN);
    assertThat(compactInfo.isImmutable()).isFalse();
    SkylarkInfo mapInfo =
        new MapBackedSkylarkInfo(provider, ImmutableMap.of("f1", 5), Location.BUILTIN);
    assertThat(mapInfo.isImmutable()).isFalse();
    provider.export(Label.create("package", "target"), "provider");
    assertThat(compactInfo.isImmutable()).isTrue();
    assertThat(mapInfo.isImmutable()).isTrue();
    compactInfo =
        new CompactSkylarkInfo(provider, layout, new Object[] {5, new Object()}, Location.BUILTIN);
    assertThat(compactInfo.isImmutable()).isFalse();
    mapInfo =
        new MapBackedSkylarkInfo(
            provider, ImmutableMap.of("f1", 5, "f2", new Object()), Location.BUILTIN);
    assertThat(mapInfo.isImmutable()).isFalse();
  }

  @Test
  public void equality() throws Exception {
    Provider provider1 =
        SkylarkProvider.createUnexportedSchemaful(ImmutableList.of("f1", "f2"), Location.BUILTIN);
    Provider provider2 =
        SkylarkProvider.createUnexportedSchemaful(ImmutableList.of("f1", "f2"), Location.BUILTIN);
    ImmutableMap<String, Integer> layout = ImmutableMap.of("f1", 0, "f2", 1);
    new EqualsTester()
        .addEqualityGroup(
            new CompactSkylarkInfo(provider1, layout, new Object[] {4, null}, Location.BUILTIN),
            new MapBackedSkylarkInfo(provider1, ImmutableMap.of("f1", 4), Location.BUILTIN))
        .addEqualityGroup(
            new CompactSkylarkInfo(provider2, layout, new Object[] {4, null}, Location.BUILTIN),
            new MapBackedSkylarkInfo(provider2, ImmutableMap.of("f1", 4), Location.BUILTIN))
        .addEqualityGroup(
            new CompactSkylarkInfo(provider1, layout, new Object[] {4, 5}, Location.BUILTIN),
            new MapBackedSkylarkInfo(
                provider1, ImmutableMap.of("f1", 4, "f2", 5), Location.BUILTIN))
        .testEquals();
  }

  @Test
  public void heterogeneousConcatenation() throws Exception {
    Provider provider =
        SkylarkProvider.createUnexportedSchemaful(ImmutableList.of("f1", "f2"), Location.BUILTIN);
    ImmutableMap<String, Integer> layout = ImmutableMap.of("f1", 0, "f2", 1);
    SkylarkInfo p1 = new MapBackedSkylarkInfo(provider, ImmutableMap.of("f1", 4), Location.BUILTIN);
    CompactSkylarkInfo p2 =
        new CompactSkylarkInfo(provider, layout, new Object[] {null, 5}, Location.BUILTIN);
    Concatable result = p1.getConcatter().concat(p1, p2, Location.BUILTIN);
    assertThat(result).isInstanceOf(MapBackedSkylarkInfo.class);
    assertThat(((SkylarkInfo) result).getKeys()).containsExactly("f1", "f2");
    assertThat(((SkylarkInfo) result).getValue("f1")).isEqualTo(4);
    assertThat(((SkylarkInfo) result).getValue("f2")).isEqualTo(5);
  }

  @Test
  public void compactConcatenationReturnsCompact() throws Exception {
    Provider provider =
        SkylarkProvider.createUnexportedSchemaful(ImmutableList.of("f1", "f2"), Location.BUILTIN);
    ImmutableMap<String, Integer> layout = ImmutableMap.of("f1", 0, "f2", 1);
    CompactSkylarkInfo p1 =
        new CompactSkylarkInfo(provider, layout, new Object[] {4, null}, Location.BUILTIN);
    CompactSkylarkInfo p2 =
        new CompactSkylarkInfo(provider, layout, new Object[] {null, 5}, Location.BUILTIN);
    Concatable result = p1.getConcatter().concat(p1, p2, Location.BUILTIN);
    assertThat(result).isInstanceOf(CompactSkylarkInfo.class);
    assertThat(((CompactSkylarkInfo) result).getKeys()).containsExactly("f1", "f2");
    assertThat(((CompactSkylarkInfo) result).getValue("f1")).isEqualTo(4);
    assertThat(((CompactSkylarkInfo) result).getValue("f2")).isEqualTo(5);
  }
}
