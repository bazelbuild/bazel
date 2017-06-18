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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TransitiveInfoProviderMapOffsetBased}. */
@RunWith(JUnit4.class)
public final class TransitiveInfoProviderMapOffsetBasedTest {

  static class ProviderA implements TransitiveInfoProvider {}

  static class ProviderB implements TransitiveInfoProvider {}

  static class ProviderC implements TransitiveInfoProvider {}

  @Test
  public void testTransitiveInfoProviderMap() throws Exception {
    ProviderA providerA = new ProviderA();
    ProviderB providerB = new ProviderB();
    ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers =
        ImmutableMap.<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>builder()
            .put(ProviderA.class, providerA)
            .put(ProviderB.class, providerB)
            .build();
    TransitiveInfoProviderMapOffsetBased map = new TransitiveInfoProviderMapOffsetBased(providers);

    assertThat(map.getProvider(ProviderA.class)).isSameAs(providerA);
    assertThat(map.getProvider(ProviderB.class)).isSameAs(providerB);
    assertThat(map.getProvider(ProviderC.class)).isNull();

    ImmutableMap.Builder<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
        containedProviders =
            ImmutableMap.<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>builder();
    for (int i = 0; i < map.getProviderCount(); ++i) {
      containedProviders.put(map.getProviderClassAt(i), map.getProviderAt(i));
    }
    assertThat(containedProviders.build()).isEqualTo(providers);
  }
}
