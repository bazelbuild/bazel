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

package com.google.devtools.build.lib.rules.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link Platform}. */
@RunWith(JUnit4.class)
public class PlatformTest extends BuildViewTestCase {

  @Rule public ExpectedException expectedException = ExpectedException.none();

  @Test
  // TODO(https://github.com/bazelbuild/bazel/issues/6849): Remove this test when the functionality
  // is removed, but until then it still needs to be verified.
  public void testPlatform_autoconfig() throws Exception {
    useConfiguration(
        "--host_cpu=piii", "--cpu=k8", "--noincompatible_auto_configure_host_platform");

    scratch.file(
        "autoconfig/BUILD",
        "constraint_setting(name = 'cpu')",
        "constraint_value(name = 'x86_32', constraint_setting = ':cpu')",
        "constraint_value(name = 'x86_64', constraint_setting = ':cpu')",
        "constraint_value(name = 'another_cpu', constraint_setting = ':cpu')",
        "constraint_setting(name = 'os')",
        "constraint_value(name = 'linux', constraint_setting = ':os')",
        "constraint_value(name = 'another_os', constraint_setting = ':os')",
        "platform(name = 'host',",
        "    host_platform = True,",
        "    cpu_constraints = [':x86_32', 'x86_64', ':another_cpu'],",
        "    os_constraints = [':linux', ':another_os'],",
        ")",
        "platform(name = 'target',",
        "    target_platform = True,",
        "    cpu_constraints = [':x86_32', 'x86_64', ':another_cpu'],",
        "    os_constraints = [':linux', ':another_os'],",
        ")");

    // Check the host platform.
    ConfiguredTarget hostPlatform = getConfiguredTarget("//autoconfig:host");
    assertThat(hostPlatform).isNotNull();

    PlatformInfo hostPlatformProvider = PlatformProviderUtils.platform(hostPlatform);
    assertThat(hostPlatformProvider).isNotNull();

    // Check the CPU and OS.
    ConstraintSettingInfo cpuConstraint =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//autoconfig:cpu"));
    ConstraintSettingInfo osConstraint =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//autoconfig:os"));
    assertThat(hostPlatformProvider.constraints().get(cpuConstraint))
        .isEqualTo(
            ConstraintValueInfo.create(
                cpuConstraint, Label.parseAbsoluteUnchecked("//autoconfig:x86_32")));
    assertThat(hostPlatformProvider.constraints().get(osConstraint))
        .isEqualTo(
            ConstraintValueInfo.create(
                osConstraint, Label.parseAbsoluteUnchecked("//autoconfig:linux")));

    // Check the target platform.
    ConfiguredTarget targetPlatform = getConfiguredTarget("//autoconfig:target");
    assertThat(targetPlatform).isNotNull();

    PlatformInfo targetPlatformProvider = PlatformProviderUtils.platform(targetPlatform);
    assertThat(targetPlatformProvider).isNotNull();

    // Check the CPU and OS.
    assertThat(targetPlatformProvider.constraints().get(cpuConstraint))
        .isEqualTo(
            ConstraintValueInfo.create(
                cpuConstraint, Label.parseAbsoluteUnchecked("//autoconfig:x86_64")));
    assertThat(targetPlatformProvider.constraints().get(osConstraint))
        .isEqualTo(
            ConstraintValueInfo.create(
                osConstraint, Label.parseAbsoluteUnchecked("//autoconfig:linux")));
  }
}
