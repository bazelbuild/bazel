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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ResourceFilterFactory}. */
@RunWith(Enclosed.class)
public class ResourceFilterFactoryTest extends ResourceTestBase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends ResourceFilterFactoryTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends ResourceFilterFactoryTest {
    @Override
    protected boolean platformBasedToolchains() {
      return true;
    }
  }

  @Before
  public void setupCcToolchain() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

  @Test
  public void parseRuleAttributes() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "java/com/pkg",
            "pkg",
            "android_binary(",
            "    name = 'pkg',",
            "    manifest = 'AndroidManifest.xml',",
            "    resource_configuration_filters = ['en', 'es'],",
            "    densities = ['hdpi', 'ldpi'],",
            ")");

    ResourceFilterFactory rff =
        ResourceFilterFactory.fromRuleContextAndAttrs(getRuleContext(target));

    assertThat(rff.getConfigurationFilterString()).isEqualTo("en,es");
    assertThat(rff.getDensityString()).isEqualTo("hdpi,ldpi");
  }
}
