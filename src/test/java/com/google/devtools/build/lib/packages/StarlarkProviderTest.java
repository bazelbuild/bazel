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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkProvider}. */
@RunWith(JUnit4.class)
public final class StarlarkProviderTest {

  @Test
  public void unexportedProvider_Accessors() {
    StarlarkProvider provider = StarlarkProvider.createUnexportedSchemaless(/*location=*/ null);
    assertThat(provider.isExported()).isFalse();
    assertThat(provider.getName()).isEqualTo("<no name>");
    assertThat(provider.getPrintableName()).isEqualTo("<no name>");
    assertThat(provider.getErrorMessageForUnknownField("foo"))
        .isEqualTo("'struct' value has no field or method 'foo'");
    assertThat(provider.isImmutable()).isFalse();
    assertThat(Starlark.repr(provider)).isEqualTo("<provider>");
    assertThrows(
        IllegalStateException.class,
        () -> provider.getKey());
  }

  @Test
  public void exportedProvider_Accessors() throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseAbsolute("//foo:bar.bzl", ImmutableMap.of()), "prov");
    StarlarkProvider provider = StarlarkProvider.createExportedSchemaless(key, /*location=*/ null);
    assertThat(provider.isExported()).isTrue();
    assertThat(provider.getName()).isEqualTo("prov");
    assertThat(provider.getPrintableName()).isEqualTo("prov");
    assertThat(provider.getErrorMessageForUnknownField("foo"))
        .isEqualTo("'prov' value has no field or method 'foo'");
    assertThat(provider.isImmutable()).isTrue();
    assertThat(Starlark.repr(provider)).isEqualTo("<provider>");
    assertThat(provider.getKey()).isEqualTo(key);
  }

  @Test
  public void schemalessProvider_Instantiation() throws Exception {
    StarlarkProvider provider = StarlarkProvider.createUnexportedSchemaless(/*location=*/ null);
    StarlarkInfo info = instantiateWithA1B2C3(provider);
    assertHasExactlyValuesA1B2C3(info);
  }

  @Test
  public void schemafulProvider_Instantiation() throws Exception {
    StarlarkProvider provider = StarlarkProvider.createUnexportedSchemaful(
        ImmutableList.of("a", "b", "c"), /*location=*/ null);
    StarlarkInfo info = instantiateWithA1B2C3(provider);
    assertHasExactlyValuesA1B2C3(info);
  }

  @Test
  public void schemalessProvider_GetFields() throws Exception {
    StarlarkProvider provider = StarlarkProvider.createUnexportedSchemaless(/*location=*/ null);
    assertThat(provider.getFields()).isNull();
  }

  @Test
  public void schemafulProvider_GetFields() throws Exception {
    StarlarkProvider provider = StarlarkProvider.createUnexportedSchemaful(
        ImmutableList.of("a", "b", "c"), /*location=*/ null);
    assertThat(provider.getFields()).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void providerEquals() throws Exception {
    // All permutations of differing label and differing name.
    StarlarkProvider.Key keyFooA =
        new StarlarkProvider.Key(Label.parseAbsolute("//foo.bzl", ImmutableMap.of()), "provA");
    StarlarkProvider.Key keyFooB =
        new StarlarkProvider.Key(Label.parseAbsolute("//foo.bzl", ImmutableMap.of()), "provB");
    StarlarkProvider.Key keyBarA =
        new StarlarkProvider.Key(Label.parseAbsolute("//bar.bzl", ImmutableMap.of()), "provA");
    StarlarkProvider.Key keyBarB =
        new StarlarkProvider.Key(Label.parseAbsolute("//bar.bzl", ImmutableMap.of()), "provB");

    // 1 for each key, plus a duplicate for one of the keys, plus 2 that have no key.
    StarlarkProvider provFooA1 =
        StarlarkProvider.createExportedSchemaless(keyFooA, /*location=*/ null);
    StarlarkProvider provFooA2 =
        StarlarkProvider.createExportedSchemaless(keyFooA, /*location=*/ null);
    StarlarkProvider provFooB =
        StarlarkProvider.createExportedSchemaless(keyFooB, /*location=*/ null);
    StarlarkProvider provBarA =
        StarlarkProvider.createExportedSchemaless(keyBarA, /*location=*/ null);
    StarlarkProvider provBarB =
        StarlarkProvider.createExportedSchemaless(keyBarB, /*location=*/ null);
    StarlarkProvider provUnexported1 =
        StarlarkProvider.createUnexportedSchemaless(/*location=*/ null);
    StarlarkProvider provUnexported2 =
        StarlarkProvider.createUnexportedSchemaless(/*location=*/ null);

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

  /** Instantiates a {@link StarlarkInfo} with fields a=1, b=2, c=3 (and nothing else). */
  private static StarlarkInfo instantiateWithA1B2C3(StarlarkProvider provider) throws Exception {
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = new StarlarkThread(mu, StarlarkSemantics.DEFAULT);
      Object result =
          Starlark.call(
              thread,
              provider,
              /*args=*/ ImmutableList.of(),
              /*kwargs=*/ ImmutableMap.of(
                  "a", StarlarkInt.of(1), "b", StarlarkInt.of(2), "c", StarlarkInt.of(3)));
      assertThat(result).isInstanceOf(StarlarkInfo.class);
      return (StarlarkInfo) result;
    }
  }

  /** Asserts that a {@link StarlarkInfo} has fields a=1, b=2, c=3 (and nothing else). */
  private static void assertHasExactlyValuesA1B2C3(StarlarkInfo info) throws Exception {
    assertThat(info.getFieldNames()).containsExactly("a", "b", "c");
    assertThat(info.getValue("a")).isEqualTo(StarlarkInt.of(1));
    assertThat(info.getValue("b")).isEqualTo(StarlarkInt.of(2));
    assertThat(info.getValue("c")).isEqualTo(StarlarkInt.of(3));
  }
}
