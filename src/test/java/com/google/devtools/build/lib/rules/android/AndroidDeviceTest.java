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
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.InputFile;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidDevice}. */
@RunWith(JUnit4.class)
public class AndroidDeviceTest extends BuildViewTestCase {
  private static final String SYSTEM_IMAGE_LABEL =
      "//sdk/system_images:emulator_images_android_21_x86";
  private static final String SYSTEM_IMAGE_DIRECTORY = "sdk/system_images/android_21/x86/";
  private static final ImmutableList<String> SYSTEM_IMAGE_FILES =
      ImmutableList.of(
          SYSTEM_IMAGE_DIRECTORY + "kernel-qemu",
          SYSTEM_IMAGE_DIRECTORY + "ramdisk.img",
          SYSTEM_IMAGE_DIRECTORY + "system.img.tar.gz",
          SYSTEM_IMAGE_DIRECTORY + "userdata.img.tar.gz");
  private static final String SOURCE_PROPERTIES = SYSTEM_IMAGE_DIRECTORY + "source.properties";
  private static final String REQUIRES_KVM = "requires-kvm";

  @Before
  public void setup() throws Exception {
    scratch.file(
        "sdk/system_images/BUILD",
        "filegroup(",
        "    name = 'emulator_images_android_21_x86',",
        "    srcs = [",
        "        'android_21/x86/kernel-qemu',",
        "        'android_21/x86/ramdisk.img',",
        "        'android_21/x86/source.properties',",
        "        'android_21/x86/system.img.tar.gz',",
        "        'android_21/x86/userdata.img.tar.gz'",
        "    ],",
        ")");
    setStarlarkSemanticsOptions("--experimental_google_legacy_api");
  }

  private FilesToRunProvider getToolDependency(String label) throws Exception {
    return getHostConfiguredTarget(ruleClassProvider.getToolsRepository() + label)
        .getProvider(FilesToRunProvider.class);
  }

  private String getToolDependencyExecPathString(String label) throws Exception {
    return getToolDependency(label).getExecutable().getExecPathString();
  }

  private String getToolDependencyRunfilesPathString(String label) throws Exception {
    return getToolDependency(label).getExecutable().getRunfilesPathString();
  }

  @Test
  public void testWellFormedDevice() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "tools/android/emulated_device",
            "nexus_6",
            "android_device(",
            "    name = 'nexus_6', ",
            "    ram = 2048, ",
            "    horizontal_resolution = 720, ",
            "    vertical_resolution = 1280, ",
            "    cache = 32, ",
            "    system_image = '" + SYSTEM_IMAGE_LABEL + "',",
            "    screen_density = 280, ",
            "    vm_heap = 256",
            ")");

    Set<String> outputBasenames = new HashSet<>();
    for (Artifact outArtifact : getFilesToBuild(target).toList()) {
      outputBasenames.add(outArtifact.getPath().getBaseName());
    }

    assertWithMessage("Not generating expected outputs.")
        .that(outputBasenames)
        .containsExactly("nexus_6", "userdata_images.dat", "emulator-meta-data.pb");

    Runfiles runfiles = getDefaultRunfiles(target);
    assertThat(ActionsTestUtil.execPaths(runfiles.getArtifacts()))
        .containsAtLeast(
            getToolDependencyExecPathString("//tools/android/emulator:support_file1"),
            getToolDependencyExecPathString("//tools/android/emulator:support_file2"));

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(getFilesToBuild(target)),
                    "nexus_6_images/userdata_images.dat");

    String systemImageString = Joiner.on(" ").join(SYSTEM_IMAGE_FILES);

    Iterable<String> biosFilesExecPathStrings =
        Iterables.transform(
            getToolDependency("//tools/android/emulator:emulator_x86_bios")
                .getFilesToRun()
                .toList(),
            (artifact) -> artifact.getExecPathString());

    assertWithMessage("Invalid boot commandline.")
        .that(action.getArguments())
        .containsExactly(
            getToolDependencyExecPathString("//tools/android/emulator:unified_launcher"),
            "--action=boot",
            "--density=280",
            "--memory=2048",
            "--skin=720x1280",
            "--cache=32",
            "--vm_size=256",
            "--system_images=" + systemImageString,
            "--bios_files=" + Joiner.on(",").join(biosFilesExecPathStrings),
            "--source_properties_file=" + SOURCE_PROPERTIES,
            "--generate_output_dir="
                + targetConfig.getBinFragment()
                + "/tools/android/emulated_device/nexus_6_images",
            "--adb_static=" + getToolDependencyExecPathString("//tools/android:adb_static"),
            "--emulator_x86="
                + getToolDependencyExecPathString("//tools/android/emulator:emulator_x86"),
            "--emulator_arm="
                + getToolDependencyExecPathString("//tools/android/emulator:emulator_arm"),
            "--adb=" + getToolDependencyExecPathString("//tools/android:adb"),
            "--mksdcard=" + getToolDependencyExecPathString("//tools/android/emulator:mksd"),
            "--empty_snapshot_fs="
                + getToolDependencyExecPathString("//tools/android/emulator:empty_snapshot_fs"),
            "--flag_configured_android_tools",
            "--nocopy_system_images",
            "--single_image_file",
            "--android_sdk_path="
                + getToolDependencyExecPathString("//tools/android/emulator:sdk_path"),
            "--platform_apks=");

    assertThat(action.getExecutionInfo()).doesNotContainKey(REQUIRES_KVM);
    assertThat(ActionsTestUtil.execPaths(action.getInputs()))
        .containsAtLeast(
            getToolDependencyExecPathString("//tools/android/emulator:support_file1"),
            getToolDependencyExecPathString("//tools/android/emulator:support_file2"));

    assertThat(target.get(ExecutionInfo.PROVIDER.getKey())).isNotNull();
    ExecutionInfo executionInfo = target.get(ExecutionInfo.PROVIDER);
    assertThat(executionInfo.getExecutionInfo()).doesNotContainKey(REQUIRES_KVM);
    TemplateExpansionAction stubAction =
        (TemplateExpansionAction) getGeneratingAction(getExecutable(target));
    String stubContents = stubAction.getFileContents();
    assertThat(stubContents)
        .contains(
            String.format(
                "unified_launcher=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android/emulator:unified_launcher")));
    assertThat(stubContents)
        .contains(
            String.format(
                "adb=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android:adb")));
    assertThat(stubContents)
        .contains(
            String.format(
                "adb_static=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android:adb_static")));
    assertThat(stubContents)
        .contains(
            String.format(
                "emulator_arm=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android/emulator:emulator_arm")));
    assertThat(stubContents)
        .contains(
            String.format(
                "emulator_x86=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android/emulator:emulator_x86")));
    assertThat(stubContents)
        .contains("source_properties_file=\"${WORKSPACE_DIR}/" + SOURCE_PROPERTIES + "\"");
    assertThat(stubContents).contains("emulator_system_images=\"" + systemImageString + "\"");
  }

  @Test
  public void testWellFormedDevice_withKvm() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "tools/android/emulated_device",
            "nexus_6",
            "android_device(",
            "   name = 'nexus_6', ",
            "   ram = 2048, ",
            "   horizontal_resolution = 720, ",
            "   vertical_resolution = 1280, ",
            "   cache = 32, ",
            "   system_image = '" + SYSTEM_IMAGE_LABEL + "',",
            "   screen_density = 280, ",
            "   vm_heap = 256,",
            "   tags = ['requires-kvm']",
            ")");

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(getFilesToBuild(target)),
                    "nexus_6_images/userdata_images.dat");

    assertThat(action.getExecutionInfo()).containsEntry(REQUIRES_KVM, "");
    assertThat(target.get(ExecutionInfo.PROVIDER.getKey())).isNotNull();
    assertThat(target.get(ExecutionInfo.PROVIDER).getExecutionInfo()).containsKey(REQUIRES_KVM);
  }

  @Test
  public void testWellFormedDevice_defaultPropertiesPresent() throws Exception {
    String dummyPropPackage = "tools/android/emulated_device/data";
    String dummyPropFile = "default.properties";
    String dummyPropLabel = String.format("//%s:%s", dummyPropPackage, dummyPropFile);
    scratch.file(String.format("%s/%s", dummyPropPackage, dummyPropFile), "ro.build.id=HiThere");
    scratch.file(
        String.format("%s/BUILD", dummyPropPackage), "exports_files(['default.properties'])");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "tools/android/emulated_device",
            "nexus_6",
            "android_device(",
            "    name = 'nexus_6', ",
            "    ram = 2048, ",
            "    horizontal_resolution = 720, ",
            "    vertical_resolution = 1280, ",
            "    cache = 32, ",
            "    system_image = '" + SYSTEM_IMAGE_LABEL + "',",
            "    screen_density = 280, ",
            "    vm_heap = 256,",
            "    default_properties = '" + dummyPropLabel + "'",
            ")");

    Set<String> outputBasenames = new HashSet<>();
    for (Artifact outArtifact : getFilesToBuild(target).toList()) {
      outputBasenames.add(outArtifact.getPath().getBaseName());
    }

    assertWithMessage("Not generating expected outputs.")
        .that(outputBasenames)
        .containsExactly("nexus_6", "userdata_images.dat", "emulator-meta-data.pb");

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(getFilesToBuild(target)),
                    "nexus_6_images/userdata_images.dat");

    String systemImageString = Joiner.on(" ").join(SYSTEM_IMAGE_FILES);
    Iterable<String> biosFilesExecPathStrings =
        Iterables.transform(
            getToolDependency("//tools/android/emulator:emulator_x86_bios")
                .getFilesToRun()
                .toList(),
            Artifact::getExecPathString);

    assertWithMessage("Invalid boot commandline.")
        .that(action.getArguments())
        .containsExactly(
            getToolDependencyExecPathString("//tools/android/emulator:unified_launcher"),
            "--action=boot",
            "--density=280",
            "--memory=2048",
            "--skin=720x1280",
            "--cache=32",
            "--vm_size=256",
            "--system_images=" + systemImageString,
            "--bios_files=" + Joiner.on(",").join(biosFilesExecPathStrings),
            "--source_properties_file=" + SOURCE_PROPERTIES,
            "--generate_output_dir="
                + targetConfig.getBinFragment()
                + "/tools/android/emulated_device/nexus_6_images",
            "--adb_static=" + getToolDependencyExecPathString("//tools/android:adb_static"),
            "--emulator_x86="
                + getToolDependencyExecPathString("//tools/android/emulator:emulator_x86"),
            "--emulator_arm="
                + getToolDependencyExecPathString("//tools/android/emulator:emulator_arm"),
            "--adb=" + getToolDependencyExecPathString("//tools/android:adb"),
            "--mksdcard=" + getToolDependencyExecPathString("//tools/android/emulator:mksd"),
            "--empty_snapshot_fs="
                + getToolDependencyExecPathString("//tools/android/emulator:empty_snapshot_fs"),
            "--flag_configured_android_tools",
            "--nocopy_system_images",
            "--single_image_file",
            "--android_sdk_path="
                + getToolDependencyExecPathString("//tools/android/emulator:sdk_path"),
            "--platform_apks=",
            "--default_properties_file=" + String.format("%s/%s", dummyPropPackage, dummyPropFile));

    TemplateExpansionAction stubAction =
        (TemplateExpansionAction) getGeneratingAction(getExecutable(target));
    String stubContents = stubAction.getFileContents();
    assertThat(stubContents)
        .contains(
            String.format(
                "unified_launcher=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android/emulator:unified_launcher")));
    assertThat(stubContents)
        .contains(
            String.format(
                "adb=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android:adb")));
    assertThat(stubContents)
        .contains(
            String.format(
                "adb_static=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android:adb_static")));
    assertThat(stubContents)
        .contains(
            String.format(
                "emulator_arm=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android/emulator:emulator_arm")));
    assertThat(stubContents)
        .contains(
            String.format(
                "emulator_x86=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android/emulator:emulator_x86")));
    assertThat(stubContents)
        .contains("source_properties_file=\"${WORKSPACE_DIR}/" + SOURCE_PROPERTIES + "\"");
    assertThat(stubContents).contains("emulator_system_images=\"" + systemImageString + "\"");
    assertThat(stubContents)
        .contains(
            String.format(
                "android_sdk_path=\"${WORKSPACE_DIR}/%s\"",
                getToolDependencyRunfilesPathString("//tools/android/emulator:sdk_path")));

    assertThat(
            target.getProvider(RunfilesProvider.class).getDefaultRunfiles().getArtifacts().toList())
        .contains(getToolDependency("//tools/android/emulator:unified_launcher").getExecutable());
  }

  @Test
  public void testPlatformApksFlag_multipleApks() throws Exception {
    String dummyPlatformApkPackage = "tools/android/emulated_device/data";
    List<String> dummyPlatformApkFiles =
        Lists.newArrayList("dummy1.apk", "dummy2.apk", "dummy3.apk", "dummy4.apk");
    List<String> platformApkFullPaths = Lists.newArrayList();
    List<String> dummyPlatformApkLabels = Lists.newArrayList();
    for (String dummyPlatformApkFile : dummyPlatformApkFiles) {
      String platformApkFullPath =
          String.format("%s/%s", dummyPlatformApkPackage, dummyPlatformApkFile);
      platformApkFullPaths.add(platformApkFullPath);
      String dummyPlatformApkLabel =
          String.format("'//%s:%s'", dummyPlatformApkPackage, dummyPlatformApkFile);
      dummyPlatformApkLabels.add(dummyPlatformApkLabel);
      scratch.file(
          String.format("%s/%s", dummyPlatformApkPackage, dummyPlatformApkFile),
          dummyPlatformApkFile);
    }
    scratch.file(
        String.format("%s/BUILD", dummyPlatformApkPackage),
        "exports_files(['dummy1.apk', 'dummy2.apk', 'dummy3.apk', 'dummy4.apk'])");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "tools/android/emulated_device",
            "nexus_6",
            "android_device(",
            "    name = 'nexus_6', ",
            "    ram = 2048, ",
            "    horizontal_resolution = 720, ",
            "    vertical_resolution = 1280, ",
            "    cache = 32, ",
            "    system_image = '" + SYSTEM_IMAGE_LABEL + "',",
            "    screen_density = 280, ",
            "    vm_heap = 256,",
            "    platform_apks = [" + Joiner.on(", ").join(dummyPlatformApkLabels) + "]",
            ")");

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(getFilesToBuild(target)),
                    "nexus_6_images/userdata_images.dat");

    assertWithMessage("Missing platform_apks flag")
        .that(action.getArguments())
        .contains("--platform_apks=" + Joiner.on(",").join(platformApkFullPaths));
  }

  @Test
  public void testPlatformApksFlag() throws Exception {
    String dummyPlatformApkPackage = "tools/android/emulated_device/data";
    String dummyPlatformApkFile = "dummy.apk";
    String dummyPlatformApkLabel =
        String.format("//%s:%s", dummyPlatformApkPackage, dummyPlatformApkFile);
    scratch.file(String.format("%s/%s", dummyPlatformApkPackage, dummyPlatformApkFile), "dummyApk");
    scratch.file(
        String.format("%s/BUILD", dummyPlatformApkPackage), "exports_files(['dummy.apk'])");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "tools/android/emulated_device",
            "nexus_6",
            "android_device(",
            "    name = 'nexus_6', ",
            "    ram = 2048, ",
            "    horizontal_resolution = 720, ",
            "    vertical_resolution = 1280, ",
            "    cache = 32, ",
            "    system_image = '" + SYSTEM_IMAGE_LABEL + "',",
            "    screen_density = 280, ",
            "    vm_heap = 256,",
            "    platform_apks = ['" + dummyPlatformApkLabel + "']",
            ")");

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(getFilesToBuild(target)),
                    "nexus_6_images/userdata_images.dat");

    assertWithMessage("Missing platform_apks flag")
        .that(action.getArguments())
        .contains(
            "--platform_apks="
                + String.format("%s/%s", dummyPlatformApkPackage, dummyPlatformApkFile));
  }

  @Test
  public void testBadAttributes() throws Exception {
    checkError(
        "bad/ram",
        "bad_ram",
        "ram must be",
        "android_device(name = 'bad_ram', ",
        "               ram = -1, ",
        "               vm_heap = 24, ",
        "               cache = 123, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               screen_density = 456, ",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 800) ");
    checkError(
        "bad/vm",
        "bad_vm",
        "heap must be",
        "android_device(name = 'bad_vm', ",
        "               ram = 512, ",
        "               vm_heap = -24, ",
        "               cache = 123, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               screen_density = 456, ",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 800) ");
    checkError(
        "bad/cache",
        "bad_cache",
        "cache must be",
        "android_device(name = 'bad_cache', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = -123, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               screen_density = 456, ",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 800) ");
    checkError(
        "bad/density",
        "bad_density",
        "density must be",
        "android_device(name = 'bad_density', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = 23, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               screen_density = -456, ",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 800) ");
    checkError(
        "bad/horizontal",
        "bad_horizontal",
        "horizontal must be",
        "android_device(name = 'bad_horizontal', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = 23, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               screen_density = -456, ",
        "               horizontal_resolution = 100, ",
        "               vertical_resolution = 800) ");
    checkError(
        "bad/vertical",
        "bad_vertical",
        "vertical must be",
        "android_device(name = 'bad_vertical', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = 23, ",
        "               screen_density = -456, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 100) ");
    checkError(
        "bad/bogus_default_prop",
        "bogus_default_prop",
        "no such package",
        "android_device(name = 'bogus_default_prop', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = 23, ",
        "               screen_density = 311, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               default_properties = '//something/somewhere',",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 100) ");
    checkError(
        "bad/multi_default_prop",
        "multi_default_prop",
        "expected a single artifact",
        "android_device(name = 'multi_default_prop', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = 23, ",
        "               screen_density = 311, ",
        "               system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "               default_properties = '" + SYSTEM_IMAGE_LABEL + "',",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 100) ");
    checkError(
        "bad/filegroup",
        "bad_filegroup",
        "No source.properties",
        "filegroup(name = 'empty',",
        "          srcs = [])",
        "android_device(name = 'bad_filegroup', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = 23, ",
        "               screen_density = -456, ",
        "               system_image = ':empty',",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 100) ");
    checkError(
        "bad/filegroup_2",
        "bad_filegroup2",
        "Multiple source.properties",
        "filegroup(name = 'empty',",
        "          srcs = ['source.properties', 'foo/source.properties'])",
        "android_device(name = 'bad_filegroup2', ",
        "               ram = 512, ",
        "               vm_heap = 24, ",
        "               cache = 23, ",
        "               screen_density = -456, ",
        "               system_image = ':empty',",
        "               horizontal_resolution = 640, ",
        "               vertical_resolution = 100) ");
  }

  @Test
  public void testPackageWhitelist() throws Exception {
    ConfiguredTarget validPackageAndroidDevice =
        scratchConfiguredTarget(
            "foo",
            "nexus_6",
            "android_device(",
            "    name = 'nexus_6', ",
            "    ram = 2048, ",
            "    horizontal_resolution = 720, ",
            "    vertical_resolution = 1280, ",
            "    cache = 32, ",
            "    system_image = '" + SYSTEM_IMAGE_LABEL + "',",
            "    screen_density = 280, ",
            "    vm_heap = 256",
            ")");
    assertThat(validPackageAndroidDevice).isNotNull();
    InputFile mockedAndroidToolsBuildFile =
        (InputFile) getTarget(ruleClassProvider.getToolsRepository() + "//tools/android:BUILD");
    String mockedAndroidToolsBuildFileLocation =
        mockedAndroidToolsBuildFile.getPath().getPathString();
    String mockedAndroidToolsContent =
        scratch
            .readFile(mockedAndroidToolsBuildFileLocation)
            .replaceAll(Pattern.quote("packages = ['//...']"), "packages = ['//bar/...']");
    scratch.overwriteFile(mockedAndroidToolsBuildFileLocation, mockedAndroidToolsContent);
    invalidatePackages();
    checkError(
        "baz",
        "nexus_6",
        "The android_device rule may not be used in this package",
        "android_device(",
        "    name = 'nexus_6', ",
        "    ram = 2048, ",
        "    horizontal_resolution = 720, ",
        "    vertical_resolution = 1280, ",
        "    cache = 32, ",
        "    system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "    screen_density = 280, ",
        "    vm_heap = 256",
        ")");
  }

  @Test
  public void testAndroidDeviceBrokerInfoExposedToStarlark() throws Exception {
    scratch.file(
        "tools/android/emulated_device/BUILD",
        "android_device(",
        "   name = 'nexus_6', ",
        "   ram = 2048, ",
        "   horizontal_resolution = 720, ",
        "   vertical_resolution = 1280, ",
        "   cache = 32, ",
        "   system_image = '" + SYSTEM_IMAGE_LABEL + "',",
        "   screen_density = 280, ",
        "   vm_heap = 256,",
        "   tags = ['requires-kvm']",
        ")");
    scratch.file(
        "javatests/com/app/starlarktest/starlarktest.bzl",
        "mystring = provider(fields = ['content'])",
        "def _impl(ctx):",
        "  return [mystring(content = ctx.attr.dep[AndroidDeviceBrokerInfo])]",
        "starlarktest = rule(implementation=_impl, attrs = {'dep': attr.label()})");
    scratch.file(
        "javatests/com/app/starlarktest/BUILD",
        "load(':starlarktest.bzl', 'starlarktest')",
        "starlarktest(name = 'mytest', dep = '//tools/android/emulated_device:nexus_6')");
    ConfiguredTarget ct = getConfiguredTarget("//javatests/com/app/starlarktest:mytest");
    assertThat(ct).isNotNull();
  }
}
