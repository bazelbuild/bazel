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

package com.google.devtools.build.lib.analysis.featurecontrol;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.EqualsTester;
import com.google.devtools.common.options.Options;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the FeaturePolicyOptions. */
@RunWith(JUnit4.class)
public final class FeaturePolicyOptionsTest {

  @Test
  public void testEquality() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            Options.getDefaults(FeaturePolicyOptions.class),
            Options.getDefaults(FeaturePolicyOptions.class))
        .addEqualityGroup(
            Options.parse(
                    FeaturePolicyOptions.class,
                    "--feature_control_policy=feature=//test:rest",
                    "--feature_control_policy=future=//nest:best")
                .getOptions())
        .testEquals();
  }

  @Test
  public void testHostVersionCopiesPolicies() throws Exception {
    FeaturePolicyOptions base =
        Options.parse(
                FeaturePolicyOptions.class,
                "--feature_control_policy=feature=//test:rest",
                "--feature_control_policy=future=//nest:best")
            .getOptions();
    FeaturePolicyOptions host = base.getHost(false);
    FeaturePolicyOptions hostFallback = base.getHost(true);
    assertThat(host.policies).containsExactlyElementsIn(base.policies);
    assertThat(hostFallback.policies).containsExactlyElementsIn(base.policies);
  }
}
