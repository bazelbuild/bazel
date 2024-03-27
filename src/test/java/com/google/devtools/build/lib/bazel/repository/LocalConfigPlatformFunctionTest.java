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
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import java.util.Collection;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link LocalConfigPlatformFunction}. */
@RunWith(Enclosed.class)
public class LocalConfigPlatformFunctionTest {

  /** Parameterized tests on CPU. */
  @RunWith(Parameterized.class)
  public static class CpuConstraintTest {
    @Parameters
    public static Collection<Object[]> createInputValues() {
      return ImmutableList.of(
          // CPU value tests.
          new Object[] {CPU.X86_64, "@platforms//cpu:x86_64"},
          new Object[] {CPU.X86_32, "@platforms//cpu:x86_32"},
          new Object[] {CPU.PPC, "@platforms//cpu:ppc"},
          new Object[] {CPU.ARM, "@platforms//cpu:arm"},
          new Object[] {CPU.AARCH64, "@platforms//cpu:aarch64"},
          new Object[] {CPU.S390X, "@platforms//cpu:s390x"});
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

  /** Parameterized tests on OS. */
  @RunWith(Parameterized.class)
  public static class OsConstraintTest {
    @Parameters
    public static Collection<Object[]> createInputValues() {
      return ImmutableList.of(
          // OS value tests.
          new Object[] {OS.LINUX, "@platforms//os:linux"},
          new Object[] {OS.DARWIN, "@platforms//os:osx"},
          new Object[] {OS.FREEBSD, "@platforms//os:freebsd"},
          new Object[] {OS.OPENBSD, "@platforms//os:openbsd"},
          new Object[] {OS.WINDOWS, "@platforms//os:windows"});
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

  /** Tests on overall functionality. */
  @RunWith(JUnit4.class)
  public static class FunctionTest extends BuildViewTestCase {
    private static final ConstraintSettingInfo CPU_CONSTRAINT =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("@platforms//cpu:cpu"));
    private static final ConstraintSettingInfo OS_CONSTRAINT =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("@platforms//os:os"));

    @Before
    public void addLocalConfigPlatform()
        throws InterruptedException, IOException, AbruptExitException {
      scratch.appendFile("WORKSPACE", "local_config_platform(name='local_config_platform_test')");
      invalidatePackages();
    }

    @Test
    public void generateConfigRepository() throws Exception {
      // Verify the package was created as expected.
      ConfiguredTarget hostPlatform = getConfiguredTarget("@local_config_platform_test//:host");
      assertThat(hostPlatform).isNotNull();

      PlatformInfo hostPlatformProvider = PlatformProviderUtils.platform(hostPlatform);
      assertThat(hostPlatformProvider).isNotNull();

      // Verify the OS and CPU constraints.
      ConstraintValueInfo expectedCpuConstraint =
          ConstraintValueInfo.create(
              CPU_CONSTRAINT,
              Label.parseCanonicalUnchecked(
                  LocalConfigPlatformFunction.cpuToConstraint(CPU.getCurrent())));
      assertThat(hostPlatformProvider.constraints().has(CPU_CONSTRAINT)).isTrue();
      assertThat(hostPlatformProvider.constraints().get(CPU_CONSTRAINT))
          .isEqualTo(expectedCpuConstraint);

      ConstraintValueInfo expectedOsConstraint =
          ConstraintValueInfo.create(
              OS_CONSTRAINT,
              Label.parseCanonicalUnchecked(
                  LocalConfigPlatformFunction.osToConstraint(OS.getCurrent())));
      assertThat(hostPlatformProvider.constraints().has(OS_CONSTRAINT)).isTrue();
      assertThat(hostPlatformProvider.constraints().get(OS_CONSTRAINT))
          .isEqualTo(expectedOsConstraint);
    }

    @Test
    public void testHostConstraints() throws Exception {
      scratch.file(
          "test/platform/my_platform.bzl",
          "def _impl(ctx):",
          "  constraints = [val[platform_common.ConstraintValueInfo] "
              + "for val in ctx.attr.constraints]",
          "  platform = platform_common.PlatformInfo(",
          "      label = ctx.label, constraint_values = constraints)",
          "  return [platform]",
          "my_platform = rule(",
          "  implementation = _impl,",
          "  attrs = {",
          "    'constraints': attr.label_list(providers = [platform_common.ConstraintValueInfo])",
          "  }",
          ")");
      scratch.file(
          "test/platform/BUILD",
          """
          load("@local_config_platform_test//:constraints.bzl", "HOST_CONSTRAINTS")
          load("//test/platform:my_platform.bzl", "my_platform")

          my_platform(
              name = "custom",
              constraints = HOST_CONSTRAINTS,
          )
          """);

      setBuildLanguageOptions("--experimental_platforms_api");
      ConfiguredTarget platform = getConfiguredTarget("//test/platform:custom");
      assertThat(platform).isNotNull();

      PlatformInfo provider = PlatformProviderUtils.platform(platform);
      assertThat(provider.constraints()).isNotNull();
    }
  }
}
