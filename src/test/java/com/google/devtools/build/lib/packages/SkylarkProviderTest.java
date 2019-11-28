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
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SkylarkProvider}. */
@RunWith(JUnit4.class)
public final class SkylarkProviderTest {

  @Test
  public void unexportedProvider_Accessors() {
    SkylarkProvider provider = SkylarkProvider.createUnexportedSchemaless(/*location=*/ null);
    assertThat(provider.isExported()).isFalse();
    assertThat(provider.getName()).isEqualTo("<no name>");
    assertThat(provider.getPrintableName()).isEqualTo("<no name>");
    assertThat(provider.getErrorMessageFormatForUnknownField())
        .isEqualTo("Object has no '%s' attribute.");
    assertThat(provider.isImmutable()).isFalse();
    assertThat(Starlark.repr(provider)).isEqualTo("<provider>");
    assertThrows(
        IllegalStateException.class,
        () -> provider.getKey());
  }

  @Test
  public void exportedProvider_Accessors() throws Exception {
    SkylarkKey key =
        new SkylarkKey(Label.parseAbsolute("//foo:bar.bzl", ImmutableMap.of()), "prov");
    SkylarkProvider provider = SkylarkProvider.createExportedSchemaless(key, /*location=*/ null);
    assertThat(provider.isExported()).isTrue();
    assertThat(provider.getName()).isEqualTo("prov");
    assertThat(provider.getPrintableName()).isEqualTo("prov");
    assertThat(provider.getErrorMessageFormatForUnknownField())
        .isEqualTo("'prov' object has no attribute '%s'");
    assertThat(provider.isImmutable()).isTrue();
    assertThat(Starlark.repr(provider)).isEqualTo("<provider>");
    assertThat(provider.getKey()).isEqualTo(key);
  }

  @Test
  public void schemalessProvider_Instantiation() throws Exception {
    SkylarkProvider provider = SkylarkProvider.createUnexportedSchemaless(/*location=*/ null);
    SkylarkInfo info = instantiateWithA1B2C3(provider);
    assertThat(info.isCompact()).isFalse();
    assertHasExactlyValuesA1B2C3(info);
  }

  @Test
  public void schemafulProvider_Instantiation() throws Exception {
    SkylarkProvider provider = SkylarkProvider.createUnexportedSchemaful(
        ImmutableList.of("a", "b", "c"), /*location=*/ null);
    SkylarkInfo info = instantiateWithA1B2C3(provider);
    assertThat(info.isCompact()).isTrue();
    assertHasExactlyValuesA1B2C3(info);
  }

  @Test
  public void schemalessProvider_GetFields() throws Exception {
    SkylarkProvider provider = SkylarkProvider.createUnexportedSchemaless(/*location=*/ null);
    assertThat(provider.getFields()).isNull();
  }

  @Test
  public void schemafulProvider_GetFields() throws Exception {
    SkylarkProvider provider = SkylarkProvider.createUnexportedSchemaful(
        ImmutableList.of("a", "b", "c"), /*location=*/ null);
    assertThat(provider.getFields()).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void providerEquals() throws Exception {
    // All permutations of differing label and differing name.
    SkylarkKey keyFooA =
        new SkylarkKey(Label.parseAbsolute("//foo.bzl", ImmutableMap.of()), "provA");
    SkylarkKey keyFooB =
        new SkylarkKey(Label.parseAbsolute("//foo.bzl", ImmutableMap.of()), "provB");
    SkylarkKey keyBarA =
        new SkylarkKey(Label.parseAbsolute("//bar.bzl", ImmutableMap.of()), "provA");
    SkylarkKey keyBarB =
        new SkylarkKey(Label.parseAbsolute("//bar.bzl", ImmutableMap.of()), "provB");

    // 1 for each key, plus a duplicate for one of the keys, plus 2 that have no key.
    SkylarkProvider provFooA1 =
        SkylarkProvider.createExportedSchemaless(keyFooA, /*location=*/ null);
    SkylarkProvider provFooA2 =
        SkylarkProvider.createExportedSchemaless(keyFooA, /*location=*/ null);
    SkylarkProvider provFooB =
        SkylarkProvider.createExportedSchemaless(keyFooB, /*location=*/ null);
    SkylarkProvider provBarA =
        SkylarkProvider.createExportedSchemaless(keyBarA, /*location=*/ null);
    SkylarkProvider provBarB =
        SkylarkProvider.createExportedSchemaless(keyBarB, /*location=*/ null);
    SkylarkProvider provUnexported1 =
        SkylarkProvider.createUnexportedSchemaless(/*location=*/ null);
    SkylarkProvider provUnexported2 =
        SkylarkProvider.createUnexportedSchemaless(/*location=*/ null);

    // For exported providers, different keys -> unequal, same key -> equal. For unexported
    // providers it comes down to object identity.
    new EqualsTester()
        .addEqualityGroup(provFooA1, provFooA2)
        .addEqualityGroup(provFooB)
        .addEqualityGroup(provBarA, provBarA)  // reflexive equality (exported)
        .addEqualityGroup(provBarB)
        .addEqualityGroup(provUnexported1, provUnexported1)  // reflexive equality (unexported)
        .addEqualityGroup(provUnexported2)
        .testEquals();
  }

  /** Instantiates a {@link SkylarkInfo} with fields a=1, b=2, c=3 (and nothing else). */
  private static SkylarkInfo instantiateWithA1B2C3(SkylarkProvider provider) throws Exception{
    // Code under test begins with the entry point in BaseFunction.
    StarlarkThread thread =
        StarlarkThread.builder(Mutability.create("test")).useDefaultSemantics().build();
    Object result =
        provider.call(
            ImmutableList.of(), ImmutableMap.of("a", 1, "b", 2, "c", 3), /*ast=*/ null, thread);
    assertThat(result).isInstanceOf(SkylarkInfo.class);
    return (SkylarkInfo) result;
  }

  /** Asserts that a {@link SkylarkInfo} has fields a=1, b=2, c=3 (and nothing else). */
  private static void assertHasExactlyValuesA1B2C3(SkylarkInfo info) throws Exception {
    assertThat(info.getFieldNames()).containsExactly("a", "b", "c");
    assertThat(info.getValue("a")).isEqualTo(1);
    assertThat(info.getValue("b")).isEqualTo(2);
    assertThat(info.getValue("c")).isEqualTo(3);
  }
}
