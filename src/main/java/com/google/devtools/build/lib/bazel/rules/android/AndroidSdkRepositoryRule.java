// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceConfiguredTargetFactory;

/**
 * Definition of the {@code android_sdk_repository} rule.
 */
public class AndroidSdkRepositoryRule implements RuleDefinition {
  public static final String NAME = "android_sdk_repository";

  private static final ImmutableMap<String, Label> calculateBindings(Rule rule) {
    String prefix = "@" + rule.getName() + "//:";
    ImmutableMap.Builder<String, Label> builder = ImmutableMap.builder();
    builder.put("android/sdk", Label.parseAbsoluteUnchecked(prefix + "sdk"));
    builder.put("android/d8_jar_import", Label.parseAbsoluteUnchecked(prefix + "d8_jar_import"));
    builder.put("android/dx_jar_import", Label.parseAbsoluteUnchecked(prefix + "dx_jar_import"));
    builder.put("android_sdk_for_testing", Label.parseAbsoluteUnchecked(prefix + "files"));
    builder.put(
        "has_androidsdk", Label.parseAbsoluteUnchecked("@bazel_tools//tools/android:always_true"));
    return builder.build();
  }

  private static final ImmutableList<String> calculateToolchainsToRegister(Rule rule) {
    return ImmutableList.of(String.format("@%s//:all", rule.getName()));
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setWorkspaceOnly()
        .setExternalBindingsFunction(AndroidSdkRepositoryRule::calculateBindings)
        .setToolchainsToRegisterFunction(AndroidSdkRepositoryRule::calculateToolchainsToRegister)
        /* <!-- #BLAZE_RULE(android_sdk_repository).ATTRIBUTE(path) -->
        An absolute or relative path to an Android SDK. Either this attribute or the
        <code>$ANDROID_HOME</code> environment variable must be set.

        <p>The Android SDK can be downloaded from
        <a href='https://developer.android.com'>the Android developer site</a>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("path", STRING).nonconfigurable("WORKSPACE rule"))
        // This is technically the directory for the build tools in $sdk/build-tools. In particular,
        // preview SDKs are in "$sdk/build-tools/x.y.z-preview", but the version is typically
        // actually "x.y.z-rcN". E.g., for 24, the directory is "$sdk/build-tools/24.0.0-preview",
        // but the version is e.g. "24 rc3". The android_sdk rule that is generated from
        // android_sdk_repository would need the real version ("24 rc3").
        /* <!-- #BLAZE_RULE(android_sdk_repository).ATTRIBUTE(build_tools_version) -->
        The version of the Android build tools to use from within the Android SDK. If not specified,
        the latest build tools version installed will be used.

        <p>Bazel requires build tools version 26.0.1 or later.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("build_tools_version", STRING).nonconfigurable("WORKSPACE rule"))
        /* <!-- #BLAZE_RULE(android_sdk_repository).ATTRIBUTE(api_level) -->
        The Android API level to build against by default. If not specified, the highest API level
        installed will be used.

        <p>The API level used for a given build can be overridden by the <code>android_sdk</code>
        flag. <code>android_sdk_repository</code> creates an <code>android_sdk</code> target for
        each API level installed in the SDK with name <code>@androidsdk//:sdk-${level}</code>,
        whether or not this attribute is specified. For example, to build against a non-default API
        level: <code>bazel build --android_sdk=@androidsdk//:sdk-19 //java/com/example:app</code>.

        <p>To view all <code>android_sdk</code> targets generated by <code>android_sdk_repository
        </code>, you can run <code>bazel query "kind(android_sdk, @androidsdk//...)"</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("api_level", INTEGER).nonconfigurable("WORKSPACE rule"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(AndroidSdkRepositoryRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = android_sdk_repository, TYPE = OTHER, FAMILY = Android) -->

<p>Configures Bazel to use a local Android SDK to support building Android targets.</p>

<h4 id="android_sdk_repository_examples">Examples</h4>

The minimum to set up an Android SDK for Bazel is to put an <code>android_sdk_repository</code> rule
named "androidsdk" in your <code>WORKSPACE</code> file and set the <code>$ANDROID_HOME</code>
environment variable to the path of your Android SDK. Bazel will use the highest Android API level
and build tools version installed in the Android SDK by default.

<pre class="code">
android_sdk_repository(
    name = "androidsdk",
)
</pre>

<p>To ensure reproducible builds, the <code>path</code>, <code>api_level</code> and
<code>build_tools_version</code> attributes can be set to specific values. The build will fail if
the Android SDK does not have the specified API level or build tools version installed.

<pre class="code">
android_sdk_repository(
    name = "androidsdk",
    path = "./sdk",
    api_level = 19,
    build_tools_version = "25.0.0",
)
</pre>

<p>The above example also demonstrates using a workspace-relative path to the Android SDK. This is
useful if the Android SDK is part of your Bazel workspace (e.g. if it is checked into version
control).


<h4 id="android_sdk_repository_support_libraries">Support Libraries</h4>

<p>The Support Libraries are available in the Android SDK Manager as "Android Support Repository".
This is a versioned set of common Android libraries, such as the Support and AppCompat libraries,
that is packaged as a local Maven repository. <code>android_sdk_repository</code> generates Bazel
targets for each of these libraries that can be used in the dependencies of
<code>android_binary</code> and <code>android_library</code> targets.

<p>The names of the generated targets are derived from the Maven coordinates of the libraries in the
Android Support Repository, formatted as <code>@androidsdk//${group}:${artifact}-${version}</code>.
The following example shows how an <code>android_library</code> can depend on version 25.0.0 of the
v7 appcompat library.

<pre class="code">
android_library(
    name = "lib",
    srcs = glob(["*.java"]),
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    deps = ["@androidsdk//com.android.support:appcompat-v7-25.0.0"],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
