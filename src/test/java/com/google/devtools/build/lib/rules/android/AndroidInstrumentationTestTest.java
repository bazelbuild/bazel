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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationTestTest.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationTestTest.WithoutPlatforms;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Tests for {@code android_instrumentation_test}. */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidInstrumentationTestTest extends AndroidBuildViewTestCase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidInstrumentationTestTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidInstrumentationTestTest {
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
        "java/com/app/BUILD",
        "android_binary(",
        "  name = 'app',",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "android_binary(",
        "  name = 'support',",
        "  manifest = 'AndroidManifest.xml',",
        ")");
    scratch.file(
        "javatests/com/app/BUILD",
        "android_binary(",
        "  name = 'instrumentation_app',",
        "  instruments = '//java/com/app',",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "android_device_script_fixture(",
        "  name = 'device_fixture',",
        "  cmd = 'foo bar',",
        ")",
        "android_host_service_fixture(",
        "  name = 'host_fixture',",
        "  executable = '//java/com/server',",
        "  service_names = ['foo', 'bar'],",
        ")");
    scratch.file(
        "java/com/server/BUILD",
        "java_binary(",
        "  name = 'server',",
        "  main_class = 'does.not.exist',",
        "  srcs = [],",
        ")");
    scratch.file(
        "javatests/com/app/ait/BUILD",
        "android_instrumentation_test(",
        "  name = 'ait',",
        "  test_app = '//javatests/com/app:instrumentation_app',",
        "  target_device = '//tools/android/emulated_device:nexus_6',",
        "  fixtures = [",
        "    '//javatests/com/app:device_fixture',",
        "    '//javatests/com/app:host_fixture',",
        "  ],",
        "  support_apks = [",
        "    '//java/com/app:support',",
        "  ],",
        "  data = [",
        "    'foo.txt',",
        "  ],",
        ")");
    setupTargetDevice();
    setBuildLanguageOptions("--experimental_google_legacy_api");
  }

  // TODO(ajmichael): Share this with AndroidDeviceTest.java
  private void setupTargetDevice() throws Exception {
    scratch.file(
        "tools/android/emulated_device/BUILD",
        "filegroup(",
        "  name = 'emulator_images_android_21_x86',",
        "  srcs = [",
        "    'android_21/x86/kernel-qemu',",
        "    'android_21/x86/ramdisk.img',",
        "    'android_21/x86/source.properties',",
        "    'android_21/x86/system.img.tar.gz',",
        "    'android_21/x86/userdata.img.tar.gz'",
        "  ],",
        ")",
        "android_device(",
        "  name = 'nexus_6',",
        "  ram = 2047,",
        "  horizontal_resolution = 720, ",
        "  vertical_resolution = 1280, ",
        "  cache = 32, ",
        "  system_image = ':emulator_images_android_21_x86',",
        "  screen_density = 280, ",
        "  vm_heap = 256",
        ")");
  }

  @Test
  public void testTestExecutableRunfiles() throws Exception {
    ConfiguredTargetAndData androidInstrumentationTest =
        getConfiguredTargetAndData("//javatests/com/app/ait");
    List<Artifact> runfiles =
        androidInstrumentationTest
            .getConfiguredTarget()
            .getProvider(RunfilesProvider.class)
            .getDefaultRunfiles()
            .getAllArtifacts()
            .toList();
    assertThat(runfiles)
        .containsAtLeastElementsIn(
            getHostConfiguredTarget("//tools/android/emulated_device:nexus_6")
                .getProvider(RunfilesProvider.class)
                .getDefaultRunfiles()
                .getAllArtifacts()
                .toList());
    assertThat(runfiles)
        .containsAtLeastElementsIn(
            getHostConfiguredTarget("//java/com/server")
                .getProvider(RunfilesProvider.class)
                .getDefaultRunfiles()
                .getAllArtifacts()
                .toList());
    assertThat(runfiles)
        .containsAtLeastElementsIn(
            getHostConfiguredTarget(
                    androidInstrumentationTest
                        .getTarget()
                        .getAssociatedRule()
                        .getAttrDefaultValue("$test_entry_point")
                        .toString())
                .getProvider(RunfilesProvider.class)
                .getDefaultRunfiles()
                .getAllArtifacts()
                .toList());
    assertThat(runfiles)
        .containsAtLeast(
            getDeviceFixtureScript(getConfiguredTarget("//javatests/com/app:device_fixture")),
            getInstrumentationApk(getConfiguredTarget("//javatests/com/app:instrumentation_app")),
            getTargetApk(getConfiguredTarget("//javatests/com/app:instrumentation_app")),
            getConfiguredTarget("//javatests/com/app/ait:foo.txt")
                .getProvider(FileProvider.class)
                .getFilesToBuild()
                .getSingleton());
  }

  @Test
  public void testTestExecutableContents() throws Exception {
    ConfiguredTarget androidInstrumentationTest = getConfiguredTarget("//javatests/com/app/ait");
    assertThat(androidInstrumentationTest).isNotNull();

    String testExecutableScript = getTestStubContents(androidInstrumentationTest);

    assertThat(testExecutableScript)
        .contains("instrumentation_apk=\"javatests/com/app/instrumentation_app.apk\"");
    assertThat(testExecutableScript).contains("target_apk=\"java/com/app/app.apk\"");
    assertThat(testExecutableScript).contains("support_apks=\"java/com/app/support.apk\"");
    assertThat(testExecutableScript)
        .contains(
            "declare -A device_script_fixtures=( "
                + "[javatests/com/app/cmd_device_fixtures/device_fixture/cmd.sh]=false,true )");
    assertThat(testExecutableScript).contains("host_service_fixture=\"java/com/server/server\"");
    assertThat(testExecutableScript).contains("host_service_fixture_services=\"foo,bar\"");
    assertThat(testExecutableScript)
        .contains("device_script=\"${WORKSPACE_DIR}/tools/android/emulated_device/nexus_6\"");
    assertThat(testExecutableScript).contains("data_deps=\"javatests/com/app/ait/foo.txt\"");
  }

  @Test
  public void testAtMostOneHostServiceFixture() throws Exception {
    checkError(
        "javatests/com/app/ait2",
        "ait",
        "android_instrumentation_test accepts at most one android_host_service_fixture",
        "android_host_service_fixture(",
        "  name = 'host_fixture',",
        "  executable = '//java/com/server',",
        "  service_names = ['foo', 'bar'],",
        ")",
        "android_instrumentation_test(",
        "  name = 'ait',",
        "  test_app = '//javatests/com/app:instrumentation_app',",
        "  target_device = '//tools/android/emulated_device:nexus_6',",
        "  fixtures = [",
        "    ':host_fixture',",
        "    '//javatests/com/app:host_fixture',",
        "  ],",
        ")");
  }

  @Test
  public void testInstrumentationBinaryIsInstrumenting() throws Exception {
    checkError(
        "javatests/com/app/instr",
        "ait",
        "The android_binary target //javatests/com/app/instr:app "
            + "is missing an 'instruments' attribute",
        "android_binary(",
        "  name = 'app',",
        "  srcs = ['a.java'],",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "android_instrumentation_test(",
        "  name = 'ait',",
        "  test_app = ':app',",
        "  target_device = '//tools/android/emulated_device:nexus_6',",
        ")");
  }

  @Test
  public void testAndroidInstrumentationTestWithStarlarkDevice() throws Exception {
    scratch.file(
        "javatests/com/app/starlarkdevice/local_adb_device.bzl",
        "def _impl(ctx):",
        "  ctx.actions.write(output=ctx.outputs.executable, content='', is_executable=True)",
        "  return [android_common.create_device_broker_info('LOCAL_ADB_SERVER')]",
        "local_adb_device = rule(implementation=_impl, executable=True)");
    scratch.file(
        "javatests/com/app/starlarkdevice/BUILD",
        "load(':local_adb_device.bzl', 'local_adb_device')",
        "local_adb_device(name = 'local_adb_device')",
        "android_instrumentation_test(",
        "  name = 'ait',",
        "  test_app = '//javatests/com/app:instrumentation_app',",
        "  target_device = ':local_adb_device',",
        ")");
    String testExecutableScript =
        getTestStubContents(getConfiguredTarget("//javatests/com/app/starlarkdevice:ait"));
    assertThat(testExecutableScript).contains("device_broker_type=\"LOCAL_ADB_SERVER\"");
  }

  private static Artifact getDeviceFixtureScript(ConfiguredTarget deviceScriptFixture) {
    return getFirstArtifactEndingWith(
        deviceScriptFixture.getProvider(FileProvider.class).getFilesToBuild(), ".sh");
  }

  private static Artifact getInstrumentationApk(ConfiguredTarget instrumentation) {
    return instrumentation.get(AndroidInstrumentationInfo.PROVIDER).getTarget().getApk();
  }

  private static Artifact getTargetApk(ConfiguredTarget instrumentation) {
    return instrumentation.get(ApkInfo.PROVIDER).getApk();
  }

  private String getTestStubContents(ConfiguredTarget androidInstrumentationTest) throws Exception {
    Action templateAction =
        getGeneratingAction(
            androidInstrumentationTest.getProvider(FilesToRunProvider.class).getExecutable());
    return ((TemplateExpansionAction) templateAction).getFileContents();
  }
}
