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
 * Definition of the {@code android_ndk_repository} rule.
 */
public class AndroidNdkRepositoryRule implements RuleDefinition {
  public static final String NAME = "android_ndk_repository";

  private static final ImmutableMap<String, Label> calculateBindings(Rule rule) {
    String prefix = "@" + rule.getName() + "//:";
    ImmutableMap.Builder<String, Label> builder = ImmutableMap.builder();
    builder.put("android/crosstool", Label.parseAbsoluteUnchecked(prefix + "default_crosstool"));
    builder.put("android_ndk_for_testing", Label.parseAbsoluteUnchecked(prefix + "files"));
    return builder.build();
  }

  private static final ImmutableList<String> calculateToolchainsToRegister(Rule rule) {
    return ImmutableList.of(String.format("@%s//:all", rule.getName()));
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setWorkspaceOnly()
        .setExternalBindingsFunction(AndroidNdkRepositoryRule::calculateBindings)
        .setToolchainsToRegisterFunction(AndroidNdkRepositoryRule::calculateToolchainsToRegister)
        /* <!-- #BLAZE_RULE(android_ndk_repository).ATTRIBUTE(path) -->
        An absolute or relative path to an Android NDK. Either this attribute or the
        <code>$ANDROID_NDK_HOME</code> environment variable must be set.

        <p>The Android NDK can be downloaded from
        <a href='https://developer.android.com/ndk/downloads/index.html'>the Android developer site
        </a>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("path", STRING).nonconfigurable("WORKSPACE rule"))
        /* <!-- #BLAZE_RULE(android_ndk_repository).ATTRIBUTE(api_level) -->
        The Android API level to build against. If not specified, the highest API level installed
        will be used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("api_level", INTEGER).nonconfigurable("WORKSPACE rule"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(AndroidNdkRepositoryRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = android_ndk_repository, TYPE = OTHER, FAMILY = Android) -->

<p>Configures Bazel to use an Android NDK to support building Android targets with native
code.

<p>Note that building for Android also requires an <code>android_sdk_repository</code> rule in your
<code>WORKSPACE</code> file.

<p>
For more information, read the <a href="https://docs.bazel.build/versions/master/android-ndk.html">
full documentation on using Android NDK with Bazel</a>.
</p>

<h4 id="android_ndk_repository_examples">Examples</h4>

<pre class="code">
android_ndk_repository(
    name = "androidndk",
)
</pre>

<p>The above example will locate your Android NDK from <code>$ANDROID_NDK_HOME</code> and detect
the highest API level that it supports.

<pre class="code">
android_ndk_repository(
    name = "androidndk",
    path = "./android-ndk-r20",
    api_level = 24,
)
</pre>

<p>The above example will use the Android NDK located inside your workspace in
<code>./android-ndk-r20</code>. It will use the API level 24 libraries when compiling your JNI
code.

<h4 id="android_ndk_repository_cpufeatures">cpufeatures</h4>

<p>The Android NDK contains the
<a href="https://developer.android.com/ndk/guides/cpu-features.html">cpufeatures library</a>
which can be used to detect a device's CPU at runtime. The following example demonstrates how to use
cpufeatures with Bazel.

<pre class="code">
# jni.cc
#include "ndk/sources/android/cpufeatures/cpu-features.h"
...
</pre>

<pre class="code">
# BUILD
cc_library(
    name = "jni",
    srcs = ["jni.cc"],
    deps = ["@androidndk//:cpufeatures"],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
