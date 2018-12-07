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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import java.util.Collection;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link LocalConfigPlatformFunction}. */
@RunWith(Enclosed.class)
public class LocalConfigPlatformFunctionTest {

  // Parameterized tests on CPU.
  @RunWith(Parameterized.class)
  public static class CpuConstraintTest {
    @Parameters
    public static Collection createInputValues() {
      return ImmutableList.of(
          // CPU value tests.
          new Object[] {CPU.X86_64, "@bazel_tools//platforms:x86_64"},
          new Object[] {CPU.X86_32, "@bazel_tools//platforms:x86_32"},
          new Object[] {CPU.PPC, "@bazel_tools//platforms:ppc"},
          new Object[] {CPU.ARM, "@bazel_tools//platforms:arm"},
          new Object[] {CPU.AARCH64, "@bazel_tools//platforms:aarch64"},
          new Object[] {CPU.S390X, "@bazel_tools//platforms:s390x"});
    }

    private final CPU testCpu;
    private final String expectedCpuConstraint;

    public CpuConstraintTest(CPU testCpu, String expectedCpuConstraint) {
      this.testCpu = testCpu;
      this.expectedCpuConstraint = expectedCpuConstraint;
    }

    @Test
    public void cpuConstraint() {
      String constraint = LocalConfigPlatformFunction.cpuToConstraint(testCpu);
      assertThat(constraint).isNotNull();
      assertThat(constraint).isEqualTo(expectedCpuConstraint);
    }

    @Test
    public void unknownCpuConstraint() {
      assertThat(LocalConfigPlatformFunction.cpuToConstraint(CPU.UNKNOWN)).isNull();
    }
  }

  // Parameterized tests on OS.
  @RunWith(Parameterized.class)
  public static class OsConstraintTest {
    @Parameters
    public static Collection createInputValues() {
      return ImmutableList.of(
          // OS value tests.
          new Object[] {OS.LINUX, "@bazel_tools//platforms:linux"},
          new Object[] {OS.DARWIN, "@bazel_tools//platforms:osx"},
          new Object[] {OS.FREEBSD, "@bazel_tools//platforms:freebsd"},
          new Object[] {OS.WINDOWS, "@bazel_tools//platforms:windows"});
    }

    private final OS testOs;
    private final String expectedOsConstraint;

    public OsConstraintTest(OS testOs, String expectedOsConstraint) {
      this.testOs = testOs;
      this.expectedOsConstraint = expectedOsConstraint;
    }

    @Test
    public void osConstraint() {
      String constraint = LocalConfigPlatformFunction.osToConstraint(testOs);
      assertThat(constraint).isNotNull();
      assertThat(constraint).isEqualTo(expectedOsConstraint);
    }

    @Test
    public void unknownOsConstraint() {
      assertThat(LocalConfigPlatformFunction.osToConstraint(OS.UNKNOWN)).isNull();
    }
  }

  // Tests on overall functionality.
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

      //    rewriteWorkspace("local_config_platform(name='local_config_platform_test')");
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
