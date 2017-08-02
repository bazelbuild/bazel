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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RequiredProviders.Builder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link RequiredProviders} class
 */
@RunWith(JUnit4.class)
public class RequiredProvidersTest {

  private static final String NO_PROVIDERS_REQUIRED = "no providers required";

  private static final class P1 {}
  private static final class P2 {}
  private static final class P3 {}

  private static final Provider P_NATIVE = new NativeProvider<Info>(Info.class, "p_native") {};

  private static final SkylarkProvider P_SKYLARK =
      new SkylarkProvider("p_skylark", Location.BUILTIN);

  static {
    try {
      P_SKYLARK.export(Label.create("foo/bar", "x.bzl"), "p_skylark");
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
  }

  private static final SkylarkProviderIdentifier ID_NATIVE =
      SkylarkProviderIdentifier.forKey(P_NATIVE.getKey());
  private static final SkylarkProviderIdentifier ID_SKYLARK =
      SkylarkProviderIdentifier.forKey(P_SKYLARK.getKey());
  private static final SkylarkProviderIdentifier ID_LEGACY =
      SkylarkProviderIdentifier.forLegacy("p_legacy");

  private static boolean satisfies(AdvertisedProviderSet providers,
      RequiredProviders requiredProviders) {
    boolean result = requiredProviders.isSatisfiedBy(providers);

    assertThat(
            requiredProviders.isSatisfiedBy(
                providers.getNativeProviders()::contains,
                providers.getSkylarkProviders()::contains))
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
            AdvertisedProviderSet.builder().addSkylark("p1").build(),
            RequiredProviders.acceptAnyBuilder().build()
        )).isTrue();
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
            AdvertisedProviderSet.builder().addSkylark("p1").build(),
            RequiredProviders.acceptNoneBuilder().build()
        )).isFalse();
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
                ImmutableSet.<Class<?>>of(P1.class),
                ImmutableSet.<Class<?>>of(P2.class)))
        .isTrue();
  }

  @Test
  public void nativeProvidersNoMatch() {
    assertThat(
            validateNative(
                AdvertisedProviderSet.builder().addNative(P3.class).build(),
                "P1 or P2",
                ImmutableSet.<Class<?>>of(P1.class),
                ImmutableSet.<Class<?>>of(P2.class)))
        .isFalse();
  }

  @Test
  public void skylarkProvidersAllMatch() {
    AdvertisedProviderSet providerSet = AdvertisedProviderSet.builder()
        .addSkylark(ID_LEGACY)
        .addSkylark(ID_NATIVE)
        .addSkylark(ID_SKYLARK)
        .build();
    assertThat(
            validateSkylark(
                providerSet,
                NO_PROVIDERS_REQUIRED,
                ImmutableSet.of(ID_LEGACY, ID_SKYLARK, ID_NATIVE)))
        .isTrue();
  }

  @Test
  public void skylarkProvidersBranchMatch() {
    assertThat(
            validateSkylark(
                AdvertisedProviderSet.builder().addSkylark(ID_LEGACY).build(),
                NO_PROVIDERS_REQUIRED,
                ImmutableSet.of(ID_LEGACY),
                ImmutableSet.of(ID_NATIVE)))
        .isTrue();
  }

  @Test
  public void skylarkProvidersNoMatch() {
    assertThat(
            validateSkylark(
                AdvertisedProviderSet.builder().addSkylark(ID_SKYLARK).build(),
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
                .addSkylarkSet(ImmutableSet.of(ID_LEGACY, ID_SKYLARK))
                .addSkylarkSet(ImmutableSet.of(ID_SKYLARK))
                .addNativeSet(ImmutableSet.of(P1.class, P2.class))
                .build()
                .getDescription())
        .isEqualTo("[P1, P2] or ['p_legacy', 'p_skylark'] or 'p_skylark'");
  }

  @SafeVarargs
  private static boolean validateNative(
      AdvertisedProviderSet providerSet, String missing, ImmutableSet<Class<?>>... sets) {
    Builder anyBuilder = RequiredProviders.acceptAnyBuilder();
    Builder noneBuilder = RequiredProviders.acceptNoneBuilder();
    for (ImmutableSet<Class<?>> set : sets) {
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
  private static boolean validateSkylark(
      AdvertisedProviderSet providerSet,
      String missing,
      ImmutableSet<SkylarkProviderIdentifier>... sets) {
    Builder anyBuilder = RequiredProviders.acceptAnyBuilder();
    Builder noneBuilder = RequiredProviders.acceptNoneBuilder();
    for (ImmutableSet<SkylarkProviderIdentifier> set : sets) {
      anyBuilder.addSkylarkSet(set);
      noneBuilder.addSkylarkSet(set);
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
