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
package com.google.devtools.build.lib.bazel.rules.android;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestTest;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Bazel-only android_local_test tests. */
@RunWith(JUnit4.class)
public class BazelAndroidLocalTestTest extends AndroidLocalTestTest {

  @Before
  @Override
  public void setUp() throws Exception {
    overwriteFile(
        "java/bar/BUILD",
        """
        java_library(
            name = "bar",
            srcs = ["S.java"],
            data = ["robolectric-deps.properties"],
        )
        """);

    overwriteFile("java/bar/foo.bzl",
        "extra_deps = ['//java/bar:bar']");
  }

  @Test
  public void testDisallowPrecompiledJars() throws Exception {
    checkError(
        "java/test",
        "dummyTest",
        // messages:
        "(expected .java, .srcjar or .properties)",
        // build file:
        String.format("%s(name = 'dummyTest',", getRuleName()),
        "    srcs = ['test.java', ':jar'])",
        "filegroup(name = 'jar',",
        "    srcs = ['lib.jar'])");
  }

  @Test
  public void testNoAndroidAllJarsPropertiesFileThrowsError() throws Exception {
    checkError("java/test",
        "test",
        "'robolectric-deps.properties' not found in the deps of the rule.",
        "android_local_test(name = 'test',",
        "    srcs = ['test.java'])");
  }

  @Override
  public void checkMainClass(ConfiguredTarget target, String targetName, boolean coverageEnabled)
      throws Exception {}

  @Override
  public void testDeployJar() throws Exception {
    // TODO(jingwen): Implement actual test.
  }

  @Override
  public void testInferredJavaPackageFromPackageName() throws Exception {
    // TODO(jingwen): Implement actual test.
  }

  @Override
  public void testFeatureFlagPolicyIsNotUsedIfFlagValuesNotUsed() throws Exception {
    // TODO(jingwen): Implement actual test.
  }

  @Override
  public void androidManifestMergerOrderAlphabeticalByConfiguration_MergeesSortedByPathInBinOrGen()
      throws Exception {}
}
