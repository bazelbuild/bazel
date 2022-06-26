// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.skyframe.TransitiveTraversalValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Basic tests for codec for {@link TransitiveTraversalValue}. */
@RunWith(JUnit4.class)
public class TransitiveTraversalValueCodecTest {

  private static final class PseudoProvider implements TransitiveInfoProvider {}

  private static final class PseudoProvider1 implements TransitiveInfoProvider {}

  @Test
  public void testCodec() throws Exception {
    RuleClassProvider ruleClassProvider = Mockito.mock(RuleClassProvider.class);
    when(ruleClassProvider.getRuleClassMap()).thenReturn(ImmutableMap.of());
    new SerializationTester(
            TransitiveTraversalValue.create(
                AdvertisedProviderSet.EMPTY, "foo_kind", /*errorMessage=*/ null),
            TransitiveTraversalValue.create(
                AdvertisedProviderSet.EMPTY, "foo_kind", /*errorMessage=*/ null),
            TransitiveTraversalValue.create(
                AdvertisedProviderSet.EMPTY, "foo_kind", /*errorMessage=*/ ""),
            TransitiveTraversalValue.create(
                AdvertisedProviderSet.create(
                    ImmutableSet.<Class<?>>of(PseudoProvider.class, PseudoProvider1.class),
                    ImmutableSet.<StarlarkProviderIdentifier>of()),
                "foo_kind",
                /*errorMessage=*/ "baz"))
        .addDependency(RuleClassProvider.class, ruleClassProvider)
        .runTests();
  }
}
