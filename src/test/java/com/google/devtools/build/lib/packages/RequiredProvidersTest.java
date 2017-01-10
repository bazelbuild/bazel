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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.RequiredProviders.Builder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link RequiredProviders} class
 */
@RunWith(JUnit4.class)
public class RequiredProvidersTest {
  private static boolean satisfies(final AdvertisedProviderSet providers,
      RequiredProviders requiredProviders) {
    boolean result = requiredProviders.isSatisfiedBy(providers);

    assertThat(requiredProviders.isSatisfiedBy(
        new Predicate<Class<?>>() {
          @Override
          public boolean apply(Class<?> aClass) {
            return providers.getNativeProviders().contains(aClass);
          }
        },
        new Predicate<SkylarkProviderIdentifier>() {
          @Override
          public boolean apply(SkylarkProviderIdentifier skylarkProviderIdentifier) {
            if (!skylarkProviderIdentifier.isLegacy()) {
              return false;
            }
            return providers.getSkylarkProviders()
                .contains(skylarkProviderIdentifier.getLegacyId());
          }
        }
    )).isEqualTo(result);
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

  private static final class P1 {}
  private static final class P2 {}
  private static final class P3 {}

  @Test
  public void nativeProvidersAllMatch() {
    AdvertisedProviderSet providerSet = AdvertisedProviderSet.builder()
        .addNative(P1.class)
        .addNative(P2.class)
        .build();
    assertThat(validateNative(providerSet, ImmutableSet.<Class<?>>of(P1.class, P2.class)))
        .isTrue();
  }

  @Test
  public void nativeProvidersBranchMatch() {
    assertThat(
        validateNative(
          AdvertisedProviderSet.builder()
              .addNative(P1.class)
              .build(),
          ImmutableSet.<Class<?>>of(P1.class),
          ImmutableSet.<Class<?>>of(P2.class)
        )).isTrue();
  }

  @Test
  public void nativeProvidersNoMatch() {
    assertThat(
        validateNative(
            AdvertisedProviderSet.builder()
                .addNative(P3.class)
                .build(),
            ImmutableSet.<Class<?>>of(P1.class),
            ImmutableSet.<Class<?>>of(P2.class)
        )).isFalse();
  }

  @Test
  public void skylarkProvidersAllMatch() {
    AdvertisedProviderSet providerSet = AdvertisedProviderSet.builder()
        .addSkylark("p1")
        .addSkylark("p2")
        .build();
    assertThat(validateSkylark(providerSet, ImmutableSet.of("p1", "p2")))
        .isTrue();
  }

  @Test
  public void skylarkProvidersBranchMatch() {
    assertThat(
        validateSkylark(
            AdvertisedProviderSet.builder()
                .addSkylark("p1")
                .build(),
            ImmutableSet.of("p1"),
            ImmutableSet.of("p2")
        )).isTrue();
  }

  @Test
  public void skylarkProvidersNoMatch() {
    assertThat(
        validateSkylark(
            AdvertisedProviderSet.builder()
                .addSkylark("p3")
                .build(),
            ImmutableSet.of("p1"),
            ImmutableSet.of("p2")
        )).isFalse();
  }

  @SafeVarargs
  private static boolean validateNative(AdvertisedProviderSet providerSet,
      ImmutableSet<Class<?>>... sets) {
    Builder anyBuilder = RequiredProviders.acceptAnyBuilder();
    Builder noneBuilder = RequiredProviders.acceptNoneBuilder();
    for (ImmutableSet<Class<?>> set : sets) {
      anyBuilder.addNativeSet(set);
      noneBuilder.addNativeSet(set);
    }
    boolean result = satisfies(providerSet, anyBuilder.build());
    assertThat(satisfies(providerSet, noneBuilder.build())).isEqualTo(result);
    return result;
  }

  @SafeVarargs
  private static boolean validateSkylark(
      AdvertisedProviderSet providerSet,
      ImmutableSet<String>... sets) {
    Builder anyBuilder = RequiredProviders.acceptAnyBuilder();
    Builder noneBuilder = RequiredProviders.acceptNoneBuilder();
    for (ImmutableSet<String> set : sets) {
      ImmutableSet<SkylarkProviderIdentifier> idSet = toIdSet(set);
      anyBuilder.addSkylarkSet(idSet);
      noneBuilder.addSkylarkSet(idSet);
    }
    boolean result = satisfies(providerSet, anyBuilder.build());
    assertThat(satisfies(providerSet, noneBuilder.build())).isEqualTo(result);
    return result;
  }

  private static ImmutableSet<SkylarkProviderIdentifier> toIdSet(ImmutableSet<String> set) {
    ImmutableSet.Builder<SkylarkProviderIdentifier> builder = ImmutableSet.builder();
    for (String id : set) {
      builder.add(SkylarkProviderIdentifier.forLegacy(id));
    }
    return builder.build();
  }

}
