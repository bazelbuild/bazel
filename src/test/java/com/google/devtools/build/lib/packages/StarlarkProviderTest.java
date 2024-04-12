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
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import java.util.Optional;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkProvider}. */
@RunWith(JUnit4.class)
public final class StarlarkProviderTest {

  @Test
  public void unexportedProvider_accessors() {
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).build();
    assertThat(provider.isExported()).isFalse();
    assertThat(provider.getName()).isEqualTo("<no name>");
    assertThat(provider.getPrintableName()).isEqualTo("<no name>");
    assertThat(provider.createRawConstructor().getName()).isEqualTo("<raw constructor>");
    assertThat(provider.getErrorMessageForUnknownField("foo"))
        .isEqualTo("'struct' value has no field or method 'foo'");
    assertThat(provider.isImmutable()).isFalse();
    assertThat(Starlark.repr(provider)).isEqualTo("<provider>");
    assertThrows(IllegalStateException.class, provider::getKey);
  }

  @Test
  public void exportedProvider_accessors() throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//foo:bar.bzl"), "prov");
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).setExported(key).build();
    assertThat(provider.isExported()).isTrue();
    assertThat(provider.getName()).isEqualTo("prov");
    assertThat(provider.getPrintableName()).isEqualTo("prov");
    assertThat(provider.createRawConstructor().getName()).isEqualTo("<raw constructor for prov>");
    assertThat(provider.getErrorMessageForUnknownField("foo"))
        .isEqualTo("'prov' value has no field or method 'foo'");
    assertThat(provider.isImmutable()).isTrue();
    assertThat(Starlark.repr(provider)).isEqualTo("<provider>");
    assertThat(provider.getKey()).isEqualTo(key);
  }

  @Test
  public void basicInstantiation() throws Exception {
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).build();
    StarlarkInfo infoFromNormalConstructor = instantiateWithA1B2C3(provider);
    assertHasExactlyValuesA1B2C3(infoFromNormalConstructor);
    assertThat(infoFromNormalConstructor.getProvider()).isEqualTo(provider);

    StarlarkInfo infoFromRawConstructor = instantiateWithA1B2C3(provider.createRawConstructor());
    assertHasExactlyValuesA1B2C3(infoFromRawConstructor);
    assertThat(infoFromRawConstructor.getProvider()).isEqualTo(provider);

    assertThat(infoFromNormalConstructor).isEqualTo(infoFromRawConstructor);
  }

  @Test
  public void instantiationWithInit() throws Exception {
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).setInit(initBC).build();
    StarlarkInfo infoFromNormalConstructor = instantiateWithA1(provider);
    assertHasExactlyValuesA1B2C3(infoFromNormalConstructor);
    assertThat(infoFromNormalConstructor.getProvider()).isEqualTo(provider);
  }

  @Test
  public void instantiationWithInitSignatureMismatch_fails() throws Exception {
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).setInit(initBC).build();
    EvalException e = assertThrows(EvalException.class, () -> instantiateWithA1B2C3(provider));
    assertThat(e).hasMessageThat().contains("expected a single `a` argument");
  }

  @Test
  public void instantiationWithInitReturnTypeMismatch_fails() throws Exception {
    StarlarkCallable initWithInvalidReturnType =
        new StarlarkCallable() {
          @Override
          public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) {
            return "invalid";
          }

          @Override
          public String getName() {
            return "initWithInvalidReturnType";
          }

          @Override
          public Location getLocation() {
            return Location.BUILTIN;
          }
        };

    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN).setInit(initWithInvalidReturnType).build();
    EvalException e = assertThrows(EvalException.class, () -> instantiateWithA1B2C3(provider));
    assertThat(e)
        .hasMessageThat()
        .contains("got string for 'return value of provider init()', want dict");
  }

  @Test
  public void instantiationWithFailingInit_fails() throws Exception {
    StarlarkCallable failingInit =
        new StarlarkCallable() {
          @Override
          public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
              throws EvalException {
            throw Starlark.errorf("failingInit fails");
          }

          @Override
          public String getName() {
            return "failingInit";
          }

          @Override
          public Location getLocation() {
            return Location.BUILTIN;
          }
        };

    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN).setInit(failingInit).build();
    EvalException e = assertThrows(EvalException.class, () -> instantiateWithA1B2C3(provider));
    assertThat(e).hasMessageThat().contains("failingInit fails");
  }

  @Test
  public void rawConstructorBypassesInit() throws Exception {
    StarlarkCallable init = mock(StarlarkCallable.class, "init");
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).setInit(init).build();
    StarlarkInfo infoFromRawConstructor = instantiateWithA1B2C3(provider.createRawConstructor());
    assertHasExactlyValuesA1B2C3(infoFromRawConstructor);
    assertThat(infoFromRawConstructor.getProvider()).isEqualTo(provider);
    verifyNoInteractions(init);
  }

  @Test
  public void basicInstantiationWithSchemaWithSomeFieldsUnset() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a", "b", "c"))
            .build();
    StarlarkInfo infoFromNormalConstructor = instantiateWithA1(provider);
    assertHasExactlyValuesA1(infoFromNormalConstructor);
    StarlarkInfo infoFromRawConstructor = instantiateWithA1(provider.createRawConstructor());
    assertHasExactlyValuesA1(infoFromRawConstructor);
  }

  @Test
  public void basicInstantiationWithSchemaWithAllFieldsSet() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a", "b", "c"))
            .build();
    StarlarkInfo infoFromNormalConstructor = instantiateWithA1B2C3(provider);
    assertHasExactlyValuesA1B2C3(infoFromNormalConstructor);
    StarlarkInfo infoFromRawConstructor = instantiateWithA1B2C3(provider.createRawConstructor());
    assertHasExactlyValuesA1B2C3(infoFromRawConstructor);
  }

  @Test
  public void basicInstantiationWithDocumentedSchema() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableMap.of("a", "Parameter a", "b", "Parameter b", "c", "Parameter c"))
            .build();
    StarlarkInfo infoFromNormalConstructor = instantiateWithA1(provider);
    assertHasExactlyValuesA1(infoFromNormalConstructor);
    StarlarkInfo infoFromRawConstructor = instantiateWithA1B2C3(provider.createRawConstructor());
    assertHasExactlyValuesA1B2C3(infoFromRawConstructor);
  }

  @Test
  public void schemaDisallowsUnexpectedFields() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN).setSchema(ImmutableList.of("a", "b")).build();
    EvalException e = assertThrows(EvalException.class, () -> instantiateWithA1B2C3(provider));
    assertThat(e)
        .hasMessageThat()
        .contains("got unexpected field 'c' in call to instantiate provider");
  }

  @Test
  public void documentedSchemaDisallowsUnexpectedFields() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableMap.of("a", "Parameter a", "b", "Parameter b"))
            .build();
    EvalException e = assertThrows(EvalException.class, () -> instantiateWithA1B2C3(provider));
    assertThat(e)
        .hasMessageThat()
        .contains("got unexpected field 'c' in call to instantiate provider");
  }

  @Test
  public void schemaEnforcedOnRawConstructor() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN).setSchema(ImmutableList.of("a", "b")).build();
    EvalException e =
        assertThrows(
            EvalException.class, () -> instantiateWithA1B2C3(provider.createRawConstructor()));
    assertThat(e)
        .hasMessageThat()
        .contains("got unexpected field 'c' in call to instantiate provider");
  }

  @Test
  public void schemaEnforcedOnInit() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a", "b"))
            .setInit(initBC)
            .build();
    EvalException e = assertThrows(EvalException.class, () -> instantiateWithA1(provider));
    assertThat(e)
        .hasMessageThat()
        .contains("got unexpected field 'c' in call to instantiate provider");
  }

  @Test
  public void documentedProvider_getDocumentation() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN).setDocumentation("My doc string").build();
    assertThat(provider.getDocumentation()).hasValue("My doc string");
  }

  @Test
  public void undocumentedProvider_getDocumentation() throws Exception {
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).build();
    assertThat(provider.getDocumentation()).isEmpty();
  }

  @Test
  public void schemalessProvider_getFields() throws Exception {
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).build();
    assertThat(provider.getFields()).isNull();
  }

  @Test
  public void schemalessProvider_getSchema() throws Exception {
    StarlarkProvider provider = StarlarkProvider.builder(Location.BUILTIN).build();
    assertThat(provider.getSchema()).isNull();
  }

  @Test
  public void providerWithUndocumentedSchema_getFields() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            // Note fields in setSchema() call below are not alphabetized to simulate
            // non-alphabetized field order in a provider declaration in Starlark code.
            .setSchema(ImmutableList.of("c", "a", "b"))
            .build();
    assertThat(provider.getFields()).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void providerWithUndocumentedSchema_getSchema() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            // Note fields in setSchema() call below are not alphabetized to simulate
            // non-alphabetized field order in a provider declaration in Starlark code.
            .setSchema(ImmutableList.of("c", "a", "b"))
            .build();
    assertThat(provider.getSchema())
        .containsExactly("c", Optional.empty(), "a", Optional.empty(), "b", Optional.empty())
        .inOrder();
  }

  @Test
  public void providerWithDocumentedSchema_getFields() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            // Note fields in setSchema() call below are not alphabetized to simulate
            // non-alphabetized field order in a provider declaration in Starlark code.
            .setSchema(ImmutableMap.of("c", "Parameter c", "a", "Parameter a", "b", "Parameter b"))
            .build();
    assertThat(provider.getFields()).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void providerWithDocumentedSchema_getSchema() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            // Note fields in setSchema() call below are not alphabetized to simulate
            // non-alphabetized field order in a provider declaration in Starlark code.
            .setSchema(ImmutableMap.of("c", "Parameter c", "a", "Parameter a", "b", "Parameter b"))
            .build();
    assertThat(provider.getSchema())
        .containsExactly(
            "c",
            Optional.of("Parameter c"),
            "a",
            Optional.of("Parameter a"),
            "b",
            Optional.of("Parameter b"))
        .inOrder();
  }

  /**
   * Tests the safe storage and retrieval of depsets, which may be optimized to nested sets in the
   * internal representation.
   */
  @Test
  public void schemafulProvider_withDepset() throws Exception {
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN).setSchema(ImmutableList.of("field")).build();
    StarlarkInfo instance1;
    StarlarkInfo instance2;
    StarlarkInfo instance3;
    StarlarkInfo instance4;
    StarlarkInfo instance5;
    StarlarkInfo instance6;
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      // Instantiates provider with values of different types all in the same field.
      // Instance with an empty depset of string
      instance1 =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of(
                      "field", Depset.of(String.class, NestedSetBuilder.emptySet(STABLE_ORDER))));
      // Instance with a non-empty depset of string
      instance2 =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of(
                      "field",
                      Depset.of(String.class, NestedSetBuilder.create(STABLE_ORDER, "foo"))));
      // Instance with a non-empty depset of int
      instance3 =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of(
                      "field",
                      Depset.of(
                          StarlarkInt.class,
                          NestedSetBuilder.create(STABLE_ORDER, StarlarkInt.of(1)))));
      // Instance with a string (not a depset)
      instance4 =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of("field", "foo"));
      // Instance with a None
      instance5 =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of("field", Starlark.NONE));
      // Instance with the field not set
      instance6 =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of());
    }

    assertThat(instance1.getValue("field")).isInstanceOf(Depset.class);
    assertThat(((Depset) instance1.getValue("field")).isEmpty()).isTrue();
    assertThat(instance2.getValue("field")).isInstanceOf(Depset.class);
    assertThat(((Depset) instance2.getValue("field")).getElementClass()).isEqualTo(String.class);
    assertThat(((Depset) instance2.getValue("field")).toList()).containsExactly("foo");
    assertThat(instance3.getValue("field")).isInstanceOf(Depset.class);
    assertThat(((Depset) instance3.getValue("field")).getElementClass())
        .isEqualTo(StarlarkInt.class);
    assertThat(((Depset) instance3.getValue("field")).toList()).containsExactly(StarlarkInt.of(1));
    assertThat(instance4.getValue("field")).isEqualTo("foo");
    assertThat(instance5.getValue("field")).isEqualTo(Starlark.NONE);
    assertThat(instance6.getValue("field")).isNull();
  }

  @Test
  public void schemafulProvider_mutable() throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//foo:bar.bzl"), "prov");
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a"))
            .setExported(key)
            .build();
    StarlarkInfo instance;
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      instance =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of("a", StarlarkList.of(mu, "x")));
      @SuppressWarnings("unchecked")
      StarlarkList<String> list = (StarlarkList<String>) instance.getValue("a");

      list.addElement("y"); // verifies the fields of the provider instance are mutable
      assertThat(instance.isImmutable()).isFalse();
    }

    @SuppressWarnings("unchecked")
    StarlarkList<String> list = (StarlarkList<String>) instance.getValue("a");
    assertThat((Iterable<?>) list).containsExactly("x", "y");
  }

  @Test
  public void schemafulProvider_immutable() throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//foo:bar.bzl"), "prov");
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a"))
            .setExported(key)
            .build();
    StarlarkInfo instance;
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      instance =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of("a", StarlarkList.of(mu, "x")));
    }

    assertThat(instance.isImmutable()).isTrue();
    @SuppressWarnings("unchecked")
    StarlarkList<String> list = (StarlarkList<String>) instance.getValue("a");
    assertThat((Iterable<?>) list).containsExactly("x");
    // verifies the fields of the frozen provider instance are immutable
    EvalException e = assertThrows(EvalException.class, () -> list.addElement("y"));
    assertThat(e).hasMessageThat().contains("trying to mutate a frozen list value");
  }

  @Test
  public void schemafulProviderWithDepset_isImmutable() throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//foo:bar.bzl"), "prov");
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a"))
            .setExported(key)
            .build();
    StarlarkInfo instance;
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      instance =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of(
                      "a", Depset.of(String.class, NestedSetBuilder.create(STABLE_ORDER, "foo"))));

      assertThat(instance.isImmutable()).isTrue();
    }
  }

  @Test
  public void schemafulProviderWithDepset_becomesImmutable() throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//foo:bar.bzl"), "prov");
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a", "b"))
            .setExported(key)
            .build();
    StarlarkInfo instance;
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      instance =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of(
                      "a",
                      Depset.of(String.class, NestedSetBuilder.create(STABLE_ORDER, "foo")),
                      "b",
                      StarlarkList.of(mu, "x")));

      assertThat(instance.isImmutable()).isFalse();
    }

    assertThat(instance.isImmutable()).isTrue();
  }

  @Test
  public void schemafulProvider_optimisedImmutable() throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//foo:bar.bzl"), "prov");
    StarlarkProvider provider =
        StarlarkProvider.builder(Location.BUILTIN)
            .setSchema(ImmutableList.of("a"))
            .setExported(key)
            .build();
    StarlarkInfo instance;
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      instance =
          (StarlarkInfo)
              Starlark.call(
                  thread,
                  provider,
                  /* args= */ ImmutableList.of(),
                  /* kwargs= */ ImmutableMap.of("a", StarlarkList.of(mu, "x")));
    }
    instance = instance.unsafeOptimizeMemoryLayout();

    assertThat(instance.isImmutable()).isTrue();
    @SuppressWarnings("unchecked")
    StarlarkList<String> list = (StarlarkList<String>) instance.getValue("a");
    assertThat((Iterable<?>) list).containsExactly("x");

    // verifies the fields of the frozen and optimised provider instance are immutable
    EvalException e = assertThrows(EvalException.class, () -> list.addElement("y"));
    assertThat(e).hasMessageThat().contains("trying to mutate a frozen list value");
  }

  @Test
  public void providerEquals() throws Exception {
    // All permutations of differing label and differing name.
    StarlarkProvider.Key keyFooA =
        new StarlarkProvider.Key(Label.parseCanonical("//foo.bzl"), "provA");
    StarlarkProvider.Key keyFooB =
        new StarlarkProvider.Key(Label.parseCanonical("//foo.bzl"), "provB");
    StarlarkProvider.Key keyBarA =
        new StarlarkProvider.Key(Label.parseCanonical("//bar.bzl"), "provA");
    StarlarkProvider.Key keyBarB =
        new StarlarkProvider.Key(Label.parseCanonical("//bar.bzl"), "provB");

    // 1 for each key, plus a duplicate for one of the keys, plus 2 that have no key.
    StarlarkProvider provFooA1 =
        StarlarkProvider.builder(Location.BUILTIN).setExported(keyFooA).build();
    StarlarkProvider provFooA2 =
        StarlarkProvider.builder(Location.BUILTIN).setExported(keyFooA).build();
    StarlarkProvider provFooB =
        StarlarkProvider.builder(Location.BUILTIN).setExported(keyFooB).build();
    StarlarkProvider provBarA =
        StarlarkProvider.builder(Location.BUILTIN).setExported(keyBarA).build();
    StarlarkProvider provBarB =
        StarlarkProvider.builder(Location.BUILTIN).setExported(keyBarB).build();
    StarlarkProvider provUnexported1 = StarlarkProvider.builder(Location.BUILTIN).build();
    StarlarkProvider provUnexported2 = StarlarkProvider.builder(Location.BUILTIN).build();

    // For exported providers, different keys -> unequal, same key -> equal. For unexported
    // providers it comes down to object identity.
    new EqualsTester()
        .addEqualityGroup(provFooA1, provFooA2)
        .addEqualityGroup(provFooB)
        .addEqualityGroup(provBarA, provBarA) // reflexive equality (exported)
        .addEqualityGroup(provBarB)
        .addEqualityGroup(provUnexported1, provUnexported1) // reflexive equality (unexported)
        .addEqualityGroup(provUnexported2)
        .testEquals();
  }

  /** Custom init equivalent to `def initBC(a): return {a:a, b:a*2, c:a*3}` */
  private static final StarlarkCallable initBC =
      new StarlarkCallable() {
        @Override
        public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
            throws EvalException {
          if (!args.isEmpty()) {
            throw Starlark.errorf("unexpected positional arguments");
          }
          if (kwargs.size() != 1 || !kwargs.containsKey("a")) {
            throw Starlark.errorf("expected a single `a` argument");
          }
          StarlarkInt a = (StarlarkInt) kwargs.get("a");
          return Dict.builder()
              .put("a", a)
              .put("b", StarlarkInt.of(a.toIntUnchecked() * 2))
              .put("c", StarlarkInt.of(a.toIntUnchecked() * 3))
              .build(Mutability.IMMUTABLE);
        }

        @Override
        public String getName() {
          return "initBC";
        }

        @Override
        public Location getLocation() {
          return Location.BUILTIN;
        }
      };

  /** Instantiates a {@link StarlarkInfo} with fields a=1 (and nothing else). */
  private static StarlarkInfo instantiateWithA1(StarlarkCallable provider) throws Exception {
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      Object result =
          Starlark.call(
              thread,
              provider,
              /*args=*/ ImmutableList.of(),
              /*kwargs=*/ ImmutableMap.of("a", StarlarkInt.of(1)));
      assertThat(result).isInstanceOf(StarlarkInfo.class);
      return (StarlarkInfo) result;
    }
  }

  /** Instantiates a {@link StarlarkInfo} with fields a=1, b=2, c=3 (and nothing else). */
  private static StarlarkInfo instantiateWithA1B2C3(StarlarkCallable provider) throws Exception {
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
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

  /** Asserts that a {@link StarlarkInfo} has field a=1 (and nothing else). */
  private static void assertHasExactlyValuesA1(StarlarkInfo info) throws Exception {
    assertThat(info.getFieldNames()).containsExactly("a");
    assertThat(info.getValue("a")).isEqualTo(StarlarkInt.of(1));
  }

  /** Asserts that a {@link StarlarkInfo} has fields a=1, b=2, c=3 (and nothing else). */
  private static void assertHasExactlyValuesA1B2C3(StarlarkInfo info) throws Exception {
    assertThat(info.getFieldNames()).containsExactly("a", "b", "c");
    assertThat(info.getValue("a")).isEqualTo(StarlarkInt.of(1));
    assertThat(info.getValue("b")).isEqualTo(StarlarkInt.of(2));
    assertThat(info.getValue("c")).isEqualTo(StarlarkInt.of(3));
  }
}
