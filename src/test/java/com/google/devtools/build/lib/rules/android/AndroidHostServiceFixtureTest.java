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

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.rules.android.AndroidHostServiceFixtureTest.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidHostServiceFixtureTest.WithoutPlatforms;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Tests for {@link AndroidHostServiceFixture}. */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidHostServiceFixtureTest extends AndroidBuildViewTestCase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidHostServiceFixtureTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidHostServiceFixtureTest {
    @Override
    protected boolean platformBasedToolchains() {
      return true;
    }
  }

  @Before
  public void setupCcToolchain() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

  @Before
  public void setup() throws Exception {
    scratch.file(
        "java/com/server/BUILD",
        "java_binary(",
        "  name = 'server',",
        "  main_class = 'does.not.exist',",
        "  srcs = [],",
        ")");
    scratch.file(
        "java/com/app/BUILD",
        "android_binary(",
        "  name = 'support',",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "genrule(",
        "  name = 'genrule',",
        "  outs = ['generated.apk'],",
        "  cmd = 'touch $(OUTS)',",
        ")");
  }

  @Test
  public void testPropagatesExecutableRunfiles() throws Exception {
    ConfiguredTarget hostServiceFixture =
        scratchConfiguredTarget(
            "javatests/com/app/BUILD",
            "fixture",
            "android_host_service_fixture(",
            "  name = 'fixture',",
            "  executable = '//java/com/server',",
            ")");
    assertThat(hostServiceFixture).isNotNull();
    assertThat(
            ActionsTestUtil.prettyArtifactNames(
                hostServiceFixture
                    .getProvider(RunfilesProvider.class)
                    .getDefaultRunfiles()
                    .getArtifacts()))
        .containsExactlyElementsIn(
            ActionsTestUtil.prettyArtifactNames(
                getConfiguredTarget("//java/com/server")
                    .getProvider(RunfilesProvider.class)
                    .getDefaultRunfiles()
                    .getArtifacts()));
  }

  @Test
  public void testProvidesServiceNames() throws Exception {
    ConfiguredTarget hostServiceFixture =
        scratchConfiguredTarget(
            "javatests/com/app/BUILD",
            "fixture",
            "android_host_service_fixture(",
            "  name = 'fixture',",
            "  executable = '//java/com/server',",
            "  service_names = ['proxy', 'echo'],",
            ")");
    assertThat(getHostServiceFixtureInfoProvider(hostServiceFixture).getServiceNames())
        .containsExactly("proxy", "echo")
        .inOrder();
  }

  @Test
  public void testProvidesSupportApks() throws Exception {
    ConfiguredTarget hostServiceFixture =
        scratchConfiguredTarget(
            "javatests/com/app/BUILD",
            "fixture",
            "android_host_service_fixture(",
            "  name = 'fixture',",
            "  executable = '//java/com/server',",
            "  service_names = ['proxy', 'echo'],",
            "  support_apks = [",
            "    '//java/com/app:support',",
            "    '//java/com/app:generated.apk',",
            "  ],",
            ")");
    assertThat(
            ActionsTestUtil.prettyArtifactNames(
                getHostServiceFixtureInfoProvider(hostServiceFixture).getSupportApks()))
        .containsExactly("java/com/app/support.apk", "java/com/app/generated.apk")
        .inOrder();
  }

  @Test
  public void testProvidesProvidesTestArgs() throws Exception {
    scratch.file(
        "javatests/com/app/BUILD",
        "android_host_service_fixture(",
        "  name = 'fixture_with_no_test_args',",
        "  executable = '//java/com/server',",
        ")",
        "android_host_service_fixture(",
        "  name = 'fixture_with_test_args',",
        "  executable = '//java/com/server',",
        "  provides_test_args = 1,",
        ")");
    assertThat(
            getHostServiceFixtureInfoProvider(
                    getConfiguredTarget("//javatests/com/app:fixture_with_no_test_args"))
                .getProvidesTestArgs())
        .isFalse();
    assertThat(
            getHostServiceFixtureInfoProvider(
                    getConfiguredTarget("//javatests/com/app:fixture_with_test_args"))
                .getProvidesTestArgs())
        .isTrue();
  }

  @Test
  public void testProvidesDaemon() throws Exception {
    scratch.file(
        "javatests/com/app/BUILD",
        "android_host_service_fixture(",
        "  name = 'no_daemon',",
        "  executable = '//java/com/server',",
        ")",
        "android_host_service_fixture(",
        "  name = 'daemon',",
        "  executable = '//java/com/server',",
        "  daemon = 1,",
        ")");
    assertThat(
            getHostServiceFixtureInfoProvider(getConfiguredTarget("//javatests/com/app:no_daemon"))
                .getDaemon())
        .isFalse();
    assertThat(
            getHostServiceFixtureInfoProvider(getConfiguredTarget("//javatests/com/app:daemon"))
                .getDaemon())
        .isTrue();
  }

  private AndroidHostServiceFixtureInfoProvider getHostServiceFixtureInfoProvider(
      ConfiguredTarget ct) throws Exception {
    return ct.get(AndroidHostServiceFixtureInfoProvider.ANDROID_HOST_SERVICE_FIXTURE_INFO);
  }
}
