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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;

import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.java.JavaSemantics;

/** Rule definition for android_device. */
public final class AndroidDeviceRule implements RuleDefinition {

  private final Class<? extends AndroidDevice> factoryClass;

  public AndroidDeviceRule(Class<? extends AndroidDevice> factoryClass) {
    this.factoryClass = factoryClass;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(android_device).IMPLICIT_OUTPUTS -->
        <ul>
          <li><code><var>name</var>_images/userdata.dat</code>:
          Contains image files and snapshots to start the emulator</li>
          <li><code><var>name</var>_images/emulator-meta-data.pb</code>:
          Contains serialized information necessary to pass on to the emulator to
          restart it.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS --> */

        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(vertical_resolution) -->
        The vertical screen resolution in pixels to emulate.
        The minimum value is 240.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("vertical_resolution", INTEGER).mandatory())
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(horizontal_resolution) -->
        The horizontal screen resolution in pixels to emulate.
        The minimum value is 240.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("horizontal_resolution", INTEGER).mandatory())
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(ram) -->
        The amount of ram in megabytes to emulate for the device.
        This is for the entire device, not just for a particular app installed on the device. The
        minimum value is 64 megabytes.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("ram", INTEGER).mandatory())
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(screen_density) -->
        The density of the emulated screen in pixels per inch.
        The minimum value of this is 30 ppi.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("screen_density", INTEGER).mandatory())
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(cache) -->
        The size in megabytes of the emulator's cache partition.
        The minimum value of this is 16 megabytes.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("cache", INTEGER).mandatory())
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(vm_heap) -->
        The size in megabytes of the virtual machine heap Android will use for each process.
        The minimum value is 16 megabytes.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("vm_heap", INTEGER).mandatory())
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(system_image) -->
        A filegroup containing the following files:
        <ul>
        <li>system.img: The system partition</li>
        <li>kernel-qemu: The Linux kernel the emulator will load</li>
        <li>ramdisk.img: The initrd image to use at boot time</li>
        <li>userdata.img: The initial userdata partition</li>
        <li>source.properties: A properties file containing information about the
        images</li>
        </ul>
        These files are part of the android sdk or provided by third parties (for
        example Intel provides x86 images).
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("system_image", LABEL).mandatory().legacyAllowAnyFileType())
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(default_properties) -->
        A single properties file to be placed in /default.prop on the emulator.
        This allows the rule author to further configure the emulator to appear more like
        a real device (In particular controlling its UserAgent strings and other
        behaviour that might cause an application or server to behave differently to
        a specific device). The properties in this file will override read only
        properties typically set by the emulator such as ro.product.model.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("default_properties", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(JavaSemantics.PROPERTIES))
        /* <!-- #BLAZE_RULE(android_device).ATTRIBUTE(platform_apks) -->
        A list of apks to be installed on the device at boot time.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("platform_apks", LABEL_LIST).legacyAllowAnyFileType())
        // Do not pregenerate oat files for tests by default unless the device
        // supports it.
        .add(attr("pregenerate_oat_files_for_tests", BOOLEAN).value(false))
        .add(
            attr("$adb_static", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android:adb_static")))
        .add(
            attr("$adb", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android:adb")))
        .add(
            attr("$emulator_arm", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android/emulator:emulator_arm")))
        .add(
            attr("$emulator_x86", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android/emulator:emulator_x86")))
        .add(
            attr("$emulator_x86_bios", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android/emulator:emulator_x86_bios")))
        .add(
            attr("$mksd", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android/emulator:mksd")))
        .add(
            attr("$empty_snapshot_fs", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android/emulator:empty_snapshot_fs")))
        .add(
            attr("$xvfb_support", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android/emulator:xvfb_support")))
        .add(
            attr("$unified_launcher", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android/emulator:unified_launcher")))
        .add(
            attr("$android_runtest", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android:android_runtest")))
        .add(
            attr("$testing_shbase", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(env.getToolsLabel("//tools/android/emulator:shbase")))
        .add(
            attr("$sdk_path", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android/emulator:sdk_path")))
        .add(
            attr("$is_executable", BOOLEAN)
                .value(true)
                .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target"))
        .add(
            Allowlist.getAttributeFromAllowlistName(AndroidDevice.ALLOWLIST_NAME)
                .value(env.getToolsLabel("//tools/android:android_device_allowlist")))
        .removeAttribute("deps")
        .removeAttribute("data")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_device")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(factoryClass)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = android_device, TYPE = OTHER, FAMILY = Android) -->

<p>This rule creates an android emulator configured with the given
  specifications. This emulator may be started via a bazel run
  command or by executing the generated script directly. It is encouraged to depend
  on existing android_device rules rather than defining your own.
</p>
<p>This rule is a suitable target for the --run_under flag to bazel test and blaze
  run.  It starts an emulator, copies the target being tested/run to the emulator,
  and tests it or runs it as appropriate.
</p>
<p><code>android_device</code> supports creating KVM images if the underlying
  <a href="${link android_device.system_image}">system_image</a> is X86 based and is
  optimized for at most the I686 CPU architecture. To use KVM add
  <code> tags = ['requires-kvm'] </code> to the <code>android_device</code> rule.
</p>

${IMPLICIT_OUTPUTS}

<h4 id="android_device_examples">Examples</h4>

<p>The following example shows how to use android_device.
<code>//java/android/helloandroid/BUILD</code> contains</p>
<pre class="code">
android_device(
    name = "nexus_s",
    cache = 32,
    default_properties = "nexus_s.properties",
    horizontal_resolution = 480,
    ram = 512,
    screen_density = 233,
    system_image = ":emulator_images_android_16_x86",
    vertical_resolution = 800,
    vm_heap = 32,
)

filegroup(
    name = "emulator_images_android_16_x86",
    srcs = glob(["androidsdk/system-images/android-16/**"]),
)
</pre>
  <p><code>//java/android/helloandroid/nexus_s.properties</code> contains:</p>
  <pre class="code">
ro.product.brand=google
ro.product.device=crespo
ro.product.manufacturer=samsung
ro.product.model=Nexus S
ro.product.name=soju
</pre>
<p>
  This rule will generate images and a start script. You can start the emulator
  locally by executing bazel run :nexus_s -- --action=start. The script exposes
  the following flags:
</p>
  <ul>
    <li>--adb_port: The port to expose adb on. If you wish to issue adb
    commands to the emulator this is the port you will issue adb connect
    to.</li>
    <li>--emulator_port: The port to expose the emulator's telnet management
    console on.</li>
    <li>--enable_display: Starts the emulator with a display if true (defaults
    to false).</li>
    <li>--action: Either start or kill.</li>
    <li>--apks_to_install: a list of apks to install on the emulator.</li>
  </ul>

<!-- #END_BLAZE_RULE -->*/
