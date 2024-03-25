// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.BazelMockAndroidSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Android Starlark APIs. */
@RunWith(JUnit4.class)
public class AndroidStarlarkTest extends AndroidBuildViewTestCase {
  @Before
  public void setUp() throws Exception {
    scratch.file(
        "java/android/platforms/BUILD",
        "platform(",
        "    name = 'x86',",
        "    parents = ['" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "android:armeabi-v7a'],",
        "    constraint_values = ['" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_32'],",
        ")",
        "platform(",
        "    name = 'armeabi-v7a',",
        "    parents = ['" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "android:armeabi-v7a'],",
        "    constraint_values = ['" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7'],",
        ")");
    scratch.file(
        "/workspace/platform_mappings",
        "flags:",
        "  --cpu=armeabi-v7a",
        "    //java/android/platforms:armeabi-v7a",
        "  --cpu=x86",
        "    //java/android/platforms:x86");
    invalidatePackages(false);
  }

  private ImmutableList<String> getPlatformLabels(ConfiguredTarget target) {
    return getConfiguration(target).getOptions().get(PlatformOptions.class).platforms.stream()
        .map(Label::toString)
        .collect(toImmutableList());
  }

  @Test
  public void testAndroidSplitTransition_android_platforms() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfigForCpu(mockToolsConfig, "x86", "armeabi-v7a");
    writeAndroidSplitTransitionTestFiles("x86");

    useConfiguration(
        "--android_platforms=//java/android/platforms:x86,//java/android/platforms:armeabi-v7a");
    ConfiguredTarget target = getConfiguredTarget("//test/starlark:test");
    StructImpl myInfo = getMyInfoFromTarget(target);

    // Check that ctx.split_attr.deps has this structure:
    // {
    //   "x86": [ConfiguredTarget],
    //   "armeabi-v7a": [ConfiguredTarget],
    // }
    @SuppressWarnings("unchecked")
    Map<String, List<ConfiguredTarget>> splitDeps =
        (Map<String, List<ConfiguredTarget>>) myInfo.getValue("split_attr_deps");
    assertThat(splitDeps).containsKey("x86");
    assertThat(splitDeps).containsKey("armeabi-v7a");
    assertThat(splitDeps.get("x86")).hasSize(2);
    assertThat(splitDeps.get("armeabi-v7a")).hasSize(2);
    assertThat(getPlatformLabels(splitDeps.get("x86").get(0)))
        .containsExactly("//java/android/platforms:x86");
    assertThat(getPlatformLabels(splitDeps.get("x86").get(1)))
        .containsExactly("//java/android/platforms:x86");

    assertThat(getPlatformLabels(splitDeps.get("armeabi-v7a").get(0)))
        .containsExactly("//java/android/platforms:armeabi-v7a");
    assertThat(getPlatformLabels(splitDeps.get("armeabi-v7a").get(1)))
        .containsExactly("//java/android/platforms:armeabi-v7a");

    // Check that ctx.split_attr.dep has this structure (that is, that the values are not lists):
    // {
    //   "x86": ConfiguredTarget,
    //   "armeabi-v7a": ConfiguredTarget,
    // }
    @SuppressWarnings("unchecked")
    Map<String, ConfiguredTarget> splitDep =
        (Map<String, ConfiguredTarget>) myInfo.getValue("split_attr_dep");
    assertThat(splitDep).containsKey("x86");
    assertThat(splitDep).containsKey("armeabi-v7a");
    assertThat(getPlatformLabels(splitDep.get("x86")))
        .containsExactly("//java/android/platforms:x86");
    assertThat(getPlatformLabels(splitDep.get("armeabi-v7a")))
        .containsExactly("//java/android/platforms:armeabi-v7a");

    // The regular ctx.attr.deps should be a single list with all the branches of the split merged
    // together (i.e. for aspects).
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> attrDeps = (List<ConfiguredTarget>) myInfo.getValue("attr_deps");
    assertThat(attrDeps).hasSize(4);
    ListMultimap<String, Object> attrDepsMap = ArrayListMultimap.create();
    for (ConfiguredTarget ct : attrDeps) {
      for (String platformLabel : getPlatformLabels(ct)) {
        attrDepsMap.put(platformLabel, target);
      }
    }
    assertThat(attrDepsMap).valuesForKey("//java/android/platforms:x86").hasSize(2);
    assertThat(attrDepsMap).valuesForKey("//java/android/platforms:armeabi-v7a").hasSize(2);

    // Check that even though my_rule.dep is defined as a single label, ctx.attr.dep is still a
    // list with multiple ConfiguredTarget objects because of the two different archs.
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> attrDep = (List<ConfiguredTarget>) myInfo.getValue("attr_dep");
    assertThat(attrDep).hasSize(2);
    ListMultimap<String, Object> attrDepMap = ArrayListMultimap.create();
    for (ConfiguredTarget ct : attrDep) {
      for (String platformLabel : getPlatformLabels(ct)) {
        attrDepMap.put(platformLabel, target);
      }
    }
    assertThat(attrDepMap).valuesForKey("//java/android/platforms:x86").hasSize(1);
    assertThat(attrDepMap).valuesForKey("//java/android/platforms:armeabi-v7a").hasSize(1);

    // Check that the deps were correctly accessed from within Starlark.
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> defaultSplitDeps =
        (List<ConfiguredTarget>) myInfo.getValue("default_split_deps");
    assertThat(defaultSplitDeps).hasSize(2);
    assertThat(getPlatformLabels(defaultSplitDeps.get(0)))
        .containsExactly("//java/android/platforms:x86");
    assertThat(getPlatformLabels(defaultSplitDeps.get(1)))
        .containsExactly("//java/android/platforms:x86");
  }

  void writeAndroidSplitTransitionTestFiles(String defaultCpuName) throws Exception {
    scratch.file(
        "test/starlark/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def impl(ctx): ",
        "  return MyInfo(",
        "    split_attr_deps = ctx.split_attr.deps,",
        "    split_attr_dep = ctx.split_attr.dep,",
        "    default_split_deps = ctx.split_attr.deps.get('" + defaultCpuName + "', None),",
        "    attr_deps = ctx.attr.deps,",
        "    attr_dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'deps': attr.label_list(cfg = android_common.multi_cpu_configuration),",
        "    'dep':  attr.label(cfg = android_common.multi_cpu_configuration),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', deps = [':main1', ':main2'], dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])",
        "cc_binary(name = 'main2', srcs = ['main2.c'])");
  }

  @Before
  public void setup() throws Exception {
    BazelMockAndroidSupport.setupNdk(mockToolsConfig);
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");
    setBuildLanguageOptions("--experimental_google_legacy_api");
  }

  StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//myinfo:myinfo.bzl"), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }
}
