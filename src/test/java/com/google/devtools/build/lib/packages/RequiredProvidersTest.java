// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link com.google.devtools.build.lib.packages.RequiredProviders} class */
@RunWith(JUnit4.class)
public class RequiredProvidersTest {

  private static final String NO_PROVIDERS_REQUIRED = "no providers required";

  private static final class P1 implements TransitiveInfoProvider {}

  private static final class P2 implements TransitiveInfoProvider {}

  private static final class P3 implements TransitiveInfoProvider {}

  private static final Provider P_NATIVE =
      new NativeProvider<StructImpl>(StructImpl.class, "p_native") {};

  private static final StarlarkProvider P_STARLARK =
      StarlarkProvider.createUnexportedSchemaless(Location.BUILTIN);

  static {
    try {
      P_STARLARK.export(Label.create("foo/bar", "x.bzl"), "p_starlark");
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
  }

  private static final StarlarkProviderIdentifier ID_NATIVE =
      StarlarkProviderIdentifier.forKey(P_NATIVE.getKey());
  private static final StarlarkProviderIdentifier ID_STARLARK =
      StarlarkProviderIdentifier.forKey(P_STARLARK.getKey());
  private static final StarlarkProviderIdentifier ID_LEGACY =
      StarlarkProviderIdentifier.forLegacy("p_legacy");

  private static boolean satisfies(AdvertisedProviderSet providers,
      RequiredProviders requiredProviders) {
    boolean result = requiredProviders.isSatisfiedBy(providers);

    assertThat(
            requiredProviders.isSatisfiedBy(
                providers.getNativeProviders()::contains,
                providers.getStarlarkProviders()::contains))
        .isEqualTo(result);
    return result;
  }

  @Test
  public void any() {
    assertThat(satisfies(AdvertisedProviderSet.EMPTY,
        RequiredProviders.acceptAnyBuilder().build())).isTrue();
    assertThat(satisfies(AdvertisedProviderSet.ANY,
        RequiredProviders.acceptAnyBuilder().build())).isTrue();
    assertThat(
        satisfies(
            AdvertisedProviderSet.builder().addNative(P1.class).build(),
            RequiredProviders.acceptAnyBuilder().build()
        )).isTrue();
    assertThat(
            satisfies(
                AdvertisedProviderSet.builder().addStarlark("p1").build(),
                RequiredProviders.acceptAnyBuilder().build()))
        .isTrue();
  }

  @Test
  public void none() {
    assertThat(satisfies(AdvertisedProviderSet.EMPTY,
        RequiredProviders.acceptNoneBuilder().build())).isFalse();
    assertThat(satisfies(AdvertisedProviderSet.ANY,
        RequiredProviders.acceptNoneBuilder().build())).isFalse();
    assertThat(
        satisfies(
            AdvertisedProviderSet.builder().addNative(P1.class).build(),
            RequiredProviders.acceptNoneBuilder().build()
        )).isFalse();
    assertThat(
            satisfies(
                AdvertisedProviderSet.builder().addStarlark("p1").build(),
                RequiredProviders.acceptNoneBuilder().build()))
        .isFalse();
  }

  @Test
  public void nativeProvidersAllMatch() {
    AdvertisedProviderSet providerSet = AdvertisedProviderSet.builder()
        .addNative(P1.class)
        .addNative(P2.class)
        .build();
    assertThat(
            validateNative(providerSet, NO_PROVIDERS_REQUIRED, ImmutableSet.of(P1.class, P2.class)))
        .isTrue();
  }

  @Test
  public void nativeProvidersBranchMatch() {
    assertThat(
            validateNative(
                AdvertisedProviderSet.builder().addNative(P1.class).build(),
                NO_PROVIDERS_REQUIRED,
                ImmutableSet.of(P1.class),
                ImmutableSet.of(P2.class)))
        .isTrue();
  }

  @Test
  public void nativeProvidersNoMatch() {
    assertThat(
            validateNative(
                AdvertisedProviderSet.builder().addNative(P3.class).build(),
                "P1 or P2",
                ImmutableSet.of(P1.class),
                ImmutableSet.of(P2.class)))
        .isFalse();
  }

  @Test
  public void starlarkProvidersAllMatch() {
    AdvertisedProviderSet providerSet =
        AdvertisedProviderSet.builder()
            .addStarlark(ID_LEGACY)
            .addStarlark(ID_NATIVE)
            .addStarlark(ID_STARLARK)
            .build();
    assertThat(
            validateStarlark(
                providerSet,
                NO_PROVIDERS_REQUIRED,
                ImmutableSet.of(ID_LEGACY, ID_STARLARK, ID_NATIVE)))
        .isTrue();
  }

  @Test
  public void starlarkProvidersBranchMatch() {
    assertThat(
            validateStarlark(
                AdvertisedProviderSet.builder().addStarlark(ID_LEGACY).build(),
                NO_PROVIDERS_REQUIRED,
                ImmutableSet.of(ID_LEGACY),
                ImmutableSet.of(ID_NATIVE)))
        .isTrue();
  }

  @Test
  public void starlarkProvidersNoMatch() {
    assertThat(
            validateStarlark(
                AdvertisedProviderSet.builder().addStarlark(ID_STARLARK).build(),
                "'p_legacy' or 'p_native'",
                ImmutableSet.of(ID_LEGACY),
                ImmutableSet.of(ID_NATIVE)))
        .isFalse();
  }

  @Test
  public void checkDescriptions() {
    assertThat(RequiredProviders.acceptAnyBuilder().build().getDescription())
        .isEqualTo("no providers required");
    assertThat(RequiredProviders.acceptNoneBuilder().build().getDescription())
        .isEqualTo("no providers accepted");
    assertThat(
            RequiredProviders.acceptAnyBuilder()
                .addStarlarkSet(ImmutableSet.of(ID_LEGACY, ID_STARLARK))
                .addStarlarkSet(ImmutableSet.of(ID_STARLARK))
                .addNativeSet(ImmutableSet.of(P1.class, P2.class))
                .build()
                .getDescription())
        .isEqualTo("[P1, P2] or ['p_legacy', 'p_starlark'] or 'p_starlark'");
  }

  @SafeVarargs
  private static boolean validateNative(
      AdvertisedProviderSet providerSet,
      String missing,
      ImmutableSet<Class<? extends TransitiveInfoProvider>>... sets) {
    RequiredProviders.Builder anyBuilder = RequiredProviders.acceptAnyBuilder();
    RequiredProviders.Builder noneBuilder = RequiredProviders.acceptNoneBuilder();
    for (ImmutableSet<Class<? extends TransitiveInfoProvider>> set : sets) {
      anyBuilder.addNativeSet(set);
      noneBuilder.addNativeSet(set);
    }
    RequiredProviders rpStartingFromAny = anyBuilder.build();
    boolean result = satisfies(providerSet, rpStartingFromAny);
    assertThat(rpStartingFromAny.getMissing(providerSet).getDescription()).isEqualTo(missing);

    RequiredProviders rpStaringFromNone = noneBuilder.build();
    assertThat(satisfies(providerSet, rpStaringFromNone)).isEqualTo(result);
    assertThat(rpStaringFromNone.getMissing(providerSet).getDescription()).isEqualTo(missing);
    return result;
  }

  @SafeVarargs
  private static boolean validateStarlark(
      AdvertisedProviderSet providerSet,
      String missing,
      ImmutableSet<StarlarkProviderIdentifier>... sets) {
    RequiredProviders.Builder anyBuilder = RequiredProviders.acceptAnyBuilder();
    RequiredProviders.Builder noneBuilder = RequiredProviders.acceptNoneBuilder();
    for (ImmutableSet<StarlarkProviderIdentifier> set : sets) {
      anyBuilder.addStarlarkSet(set);
      noneBuilder.addStarlarkSet(set);
    }

    RequiredProviders rpStartingFromAny = anyBuilder.build();
    boolean result = satisfies(providerSet, rpStartingFromAny);
    assertThat(rpStartingFromAny.getMissing(providerSet).getDescription()).isEqualTo(missing);

    RequiredProviders rpStaringFromNone = noneBuilder.build();
    assertThat(satisfies(providerSet, rpStaringFromNone)).isEqualTo(result);
    assertThat(rpStaringFromNone.getMissing(providerSet).getDescription()).isEqualTo(missing);
    return result;
  }
}
