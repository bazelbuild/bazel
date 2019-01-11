// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.posix;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Posix-only tests for {@link LocalConfigPlatformFunction}. */
@RunWith(Enclosed.class)
public class LocalConfigPlatformFunctionPosixTest {

  /** Tests on overall functionality. */
  @RunWith(JUnit4.class)
  public static class FunctionTest extends BuildViewTestCase {
    private static final ConstraintSettingInfo CPU_CONSTRAINT =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("@bazel_tools//platforms:cpu"));
    private static final ConstraintSettingInfo OS_CONSTRAINT =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("@bazel_tools//platforms:os"));

    private static final ConstraintValueInfo X86_64_CONSTRAINT =
        ConstraintValueInfo.create(
            CPU_CONSTRAINT, Label.parseAbsoluteUnchecked("@bazel_tools//platforms:x86_64"));
    private static final ConstraintValueInfo LINUX_CONSTRAINT =
        ConstraintValueInfo.create(
            OS_CONSTRAINT, Label.parseAbsoluteUnchecked("@bazel_tools//platforms:linux"));

    @Test
    public void generateConfigRepository() throws Exception {
      CPU.setForTesting(CPU.X86_64);
      OS.setForTesting(OS.LINUX);

      scratch.appendFile("WORKSPACE", "local_config_platform(name='local_config_platform_test')");
      invalidatePackages();

      // Verify the package was created as expected.
      ConfiguredTarget hostPlatform = getConfiguredTarget("@local_config_platform_test//:host");
      assertThat(hostPlatform).isNotNull();

      PlatformInfo hostPlatformProvider = PlatformProviderUtils.platform(hostPlatform);
      assertThat(hostPlatformProvider).isNotNull();

      // Verify the OS and CPU constraints.
      assertThat(hostPlatformProvider.constraints().has(CPU_CONSTRAINT)).isTrue();
      assertThat(hostPlatformProvider.constraints().get(CPU_CONSTRAINT))
          .isEqualTo(X86_64_CONSTRAINT);

      assertThat(hostPlatformProvider.constraints().has(OS_CONSTRAINT)).isTrue();
      assertThat(hostPlatformProvider.constraints().get(OS_CONSTRAINT)).isEqualTo(LINUX_CONSTRAINT);
    }

    // TODO(katre): check the host_platform_remote_properties_override flag
  }
}
