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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for the {@code android_instrumentation_test} rule. */
public class AndroidInstrumentationTestBaseRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(android_instrumentation_test).ATTRIBUTE(test_app) -->
         The <a href="${link android_binary}">android_binary</a> target containing the test classes.
         The <code>android_binary</code> target must specify which target it is testing through
         its <a href="${link android_binary.instruments}"><code>instruments</code></a> attribute.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("test_app", LABEL)
                .mandatory()
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .allowedRuleClasses("android_binary"))
        /* <!-- #BLAZE_RULE(android_instrumentation_test).ATTRIBUTE(target_device) -->
        <p>The <a href="${link android_device}">android_device</a> the test should run on.</p>
        <p>To run the test on an emulator that is already running or on a physical device, use
        these arguments:
        <code>--test_output=streamed --test_arg=--device_broker_type=LOCAL_ADB_SERVER
        --test_arg=--device_serial_number=$device_identifier</code></p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("target_device", LABEL)
                .mandatory()
                .exec()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(
                    StarlarkProviderIdentifier.forKey(AndroidDeviceBrokerInfo.PROVIDER.getKey())))
        /* <!-- #BLAZE_RULE(android_instrumentation_test).ATTRIBUTE(support_apks) -->
        Other APKs to install on the device before the instrumentation test starts.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("support_apks", LABEL_LIST)
                .allowedFileTypes(AndroidRuleClasses.APK)
                .allowedRuleClasses("android_binary"))
        .add(
            attr("fixtures", LABEL_LIST)
                .undocumented("Undocumented until the fixtures rules are documented")
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .allowedRuleClasses(
                    "android_device_script_fixture", "android_host_service_fixture"))
        .add(
            attr("$test_entry_point", LABEL)
                .exec()
                .cfg(ExecutionTransitionFactory.create())
                .value(
                    environment.getToolsLabel("//tools/android:instrumentation_test_entry_point")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$android_instrumentation_test_base")
        .type(RuleClassType.ABSTRACT)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = android_instrumentation_test, TYPE = TEST, FAMILY = Android) -->

<p>
  An <code>android_instrumentation_test</code> rule runs Android instrumentation tests. It will
  start an emulator, install the application being tested, the test application, and
  any other needed applications, and run the tests defined in the test package.
</p>
<p>
  The <a href="${link android_instrumentation_test.test_app}">test_app</a> attribute specifies the
  <code>android_binary</code> which contains the test. This <code>android_binary</code> in turn
  specifies the <code>android_binary</code> application under test through its
  <a href="${link android_binary.instruments}">instruments</a> attribute.
</p>

<h4 id="android_instrumentation_test_examples">Example</h4>

<pre class="code">
# java/com/samples/hello_world/BUILD

android_library(
    name = "hello_world_lib",
    srcs = ["Lib.java"],
    manifest = "LibraryManifest.xml",
    resource_files = glob(["res/**"]),
)

# The app under test
android_binary(
    name = "hello_world_app",
    manifest = "AndroidManifest.xml",
    deps = [":hello_world_lib"],
)
</pre>

<pre class="code">
# javatests/com/samples/hello_world/BUILD

android_library(
    name = "hello_world_test_lib",
    srcs = ["Tests.java"],
    deps = [
      "//java/com/samples/hello_world:hello_world_lib",
      ...  # test dependencies such as Espresso and Mockito
    ],
)

# The test app
android_binary(
    name = "hello_world_test_app",
    instruments = "//java/com/samples/hello_world:hello_world_app",
    manifest = "AndroidManifest.xml",
    deps = [":hello_world_test_lib"],
)

android_instrumentation_test(
    name = "hello_world_uiinstrumentation_tests",
    target_device = ":some_target_device",
    test_app = ":hello_world_test_app",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
