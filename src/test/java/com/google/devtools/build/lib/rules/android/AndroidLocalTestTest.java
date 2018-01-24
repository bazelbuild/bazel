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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for android_local_test. */
@RunWith(JUnit4.class)
public abstract class AndroidLocalTestTest extends AbstractAndroidLocalTestTestBase {

  @Test
  public void testSimpleTestNotNull() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    assertThat(target).isNotNull();
  }

  @Test
  public void testResourceFilesZipCalledResourceFilesZip() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");

    Artifact resourcesZip =
        getImplicitOutputArtifact(target, AndroidRuleClasses.ANDROID_RESOURCES_ZIP);
    assertThat(resourcesZip.getFilename()).isEqualTo("resource_files.zip");
  }

  @Test
  public void testManifestInRunfiles() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    Iterable<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact manifest =
        ActionsTestUtil.getFirstArtifactEndingWith(
            runfilesArtifacts, "dummyTest_generated/dummyTest/AndroidManifest.xml");
    assertThat(manifest).isNotNull();
  }

  @Test
  public void testResourcesClassJarInRunfiles() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    Iterable<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact resourceClassJar =
        getImplicitOutputArtifact(target, AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);
    assertThat(runfilesArtifacts).contains(resourceClassJar);
  }

  @Test
  public void testResourcesZipFileInRunfiles() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    Iterable<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact resourcesZip =
        getImplicitOutputArtifact(target, AndroidRuleClasses.ANDROID_RESOURCES_ZIP);
    assertThat(runfilesArtifacts).contains(resourcesZip);
  }

  @Test
  public void testCanHaveManifestNotNamedAndroidManifestXml() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps",
        "    manifest = 'NotAndroidManifest.xml')");
    assertNoEvents();
  }

  @Override
  protected String getRuleName() {
    return "android_local_test";
  }

  @Override
  protected void writeFile(String path, String... lines) throws Exception {
    scratch.file(path, lines);
  }

  @Override
  protected void overwriteFile(String path, String... lines) throws Exception {
    scratch.overwriteFile(path, lines);
  }
}
