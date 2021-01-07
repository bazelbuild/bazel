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
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.rules.android.AndroidDeviceScriptFixtureTest.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidDeviceScriptFixtureTest.WithoutPlatforms;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Tests for {@link AndroidDeviceScriptFixture}. */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidDeviceScriptFixtureTest extends AndroidBuildViewTestCase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidDeviceScriptFixtureTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidDeviceScriptFixtureTest {
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
    scratch.file("scripts/BUILD", "exports_files(['my_script.sh'])");
    scratch.file(
        "java/com/app/BUILD",
        "android_binary(",
        "  name = 'app',",
        "  manifest = 'AndroidManifest.xml',",
        ")");
    scratch.file(
        "java/com/app/support/BUILD",
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
  public void testScriptFixture() throws Exception {
    scratch.file(
        "javatests/com/app/BUILD",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        "  script = '//scripts:my_script.sh',",
        ")");
    ConfiguredTarget fixture = getConfiguredTarget("//javatests/com/app:fixture");
    assertThat(fixture).isNotNull();
    AndroidDeviceScriptFixtureInfoProvider deviceScriptFixtureInfoProvider =
        fixture.get(AndroidDeviceScriptFixtureInfoProvider.STARLARK_CONSTRUCTOR);
    assertThat(deviceScriptFixtureInfoProvider).isNotNull();
    assertThat(deviceScriptFixtureInfoProvider.getFixtureScript()).isNotNull();
    assertThat(deviceScriptFixtureInfoProvider.getFixtureScript().prettyPrint())
        .isEqualTo("scripts/my_script.sh");
  }

  @Test
  public void testCommandFixture() throws Exception {
    scratch.file(
        "javatests/com/app/BUILD",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        "  cmd = 'some literal command',",
        ")");
    ConfiguredTarget fixture = getConfiguredTarget("//javatests/com/app:fixture");
    assertThat(fixture).isNotNull();
    AndroidDeviceScriptFixtureInfoProvider deviceScriptFixtureInfoProvider =
        fixture.get(AndroidDeviceScriptFixtureInfoProvider.STARLARK_CONSTRUCTOR);
    assertThat(deviceScriptFixtureInfoProvider).isNotNull();
    assertThat(deviceScriptFixtureInfoProvider.getFixtureScript()).isNotNull();
    assertThat(deviceScriptFixtureInfoProvider.getFixtureScript().prettyPrint())
        .isEqualTo("javatests/com/app/cmd_device_fixtures/fixture/cmd.sh");
    FileWriteAction action =
        (FileWriteAction) getGeneratingAction(deviceScriptFixtureInfoProvider.getFixtureScript());
    assertThat(action.getFileContents()).isEqualTo("some literal command");
  }

  @Test
  public void testNoScriptOrCommand() throws Exception {
    checkError(
        "javatests/com/app",
        "fixture",
        "android_host_service_fixture requires that exactly one of the script and cmd attributes "
            + "be specified",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        ")");
  }

  @Test
  public void testScriptAndCommand() throws Exception {
    checkError(
        "javatests/com/app",
        "fixture",
        "android_host_service_fixture requires that exactly one of the script and cmd attributes "
            + "be specified",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        "  script = '//scripts:my_script.sh',",
        "  cmd = 'some literal command',",
        ")");
  }

  @Test
  public void testSupportApks() throws Exception {
    scratch.file(
        "javatests/com/app/BUILD",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        "  cmd = 'some literal command',",
        "  support_apks = [",
        "    '//java/com/app/support',",
        "    '//java/com/app/support:generated.apk',",
        "  ],",
        ")");
    ConfiguredTarget fixture = getConfiguredTarget("//javatests/com/app:fixture");
    assertThat(fixture).isNotNull();
    AndroidDeviceScriptFixtureInfoProvider deviceScriptFixtureInfoProvider =
        fixture.get(AndroidDeviceScriptFixtureInfoProvider.STARLARK_CONSTRUCTOR);
    assertThat(deviceScriptFixtureInfoProvider).isNotNull();

    assertThat(
            ActionsTestUtil.prettyArtifactNames(deviceScriptFixtureInfoProvider.getSupportApks()))
        .containsExactly("java/com/app/support/support.apk", "java/com/app/support/generated.apk")
        .inOrder();
  }

  @Test
  public void testNonSupportedScriptExtension() throws Exception {
    scratch.file("javatests/com/app/script.bat");
    checkError(
        "javatests/com/app",
        "fixture",
        "file '//javatests/com/app:script.bat' is misplaced here (expected .sh)",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        "  script = 'script.bat',",
        ")");
  }

  @Test
  public void testDaemonIsProvided() throws Exception {
    scratch.file(
        "javatests/com/app/BUILD",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        "  cmd = 'some literal command',",
        "  daemon = 1,",
        ")");
    ConfiguredTarget fixture = getConfiguredTarget("//javatests/com/app:fixture");
    assertThat(fixture).isNotNull();
    AndroidDeviceScriptFixtureInfoProvider deviceScriptFixtureInfoProvider =
        fixture.get(AndroidDeviceScriptFixtureInfoProvider.STARLARK_CONSTRUCTOR);
    assertThat(deviceScriptFixtureInfoProvider).isNotNull();
    assertThat(deviceScriptFixtureInfoProvider.getDaemon()).isTrue();
  }

  @Test
  public void testStrictExitIsProvided() throws Exception {
    scratch.file(
        "javatests/com/app/BUILD",
        "android_device_script_fixture(",
        "  name = 'fixture',",
        "  cmd = 'some literal command',",
        "  strict_exit = 1,",
        ")");
    ConfiguredTarget fixture = getConfiguredTarget("//javatests/com/app:fixture");
    assertThat(fixture).isNotNull();
    AndroidDeviceScriptFixtureInfoProvider deviceScriptFixtureInfoProvider =
        fixture.get(AndroidDeviceScriptFixtureInfoProvider.STARLARK_CONSTRUCTOR);
    assertThat(deviceScriptFixtureInfoProvider).isNotNull();
    assertThat(deviceScriptFixtureInfoProvider.getStrictExit()).isTrue();
  }
}
