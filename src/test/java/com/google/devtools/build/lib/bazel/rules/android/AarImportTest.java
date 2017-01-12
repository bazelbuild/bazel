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
package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider;
import com.google.devtools.build.lib.rules.android.NativeLibsZipsProvider;
import com.google.devtools.build.lib.rules.android.ResourceContainer;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.rules.android.AarImport}. */
@RunWith(JUnit4.class)
public class AarImportTest extends BuildViewTestCase {
  @Before
  public void setup() throws Exception {
    scratch.file("a/BUILD",
        "aar_import(",
        "    name = 'foo',",
        "    aar = 'foo.aar',",
        ")",
        "aar_import(",
        "    name = 'bar',",
        "    aar = 'bar.aar',",
        "    exports = [':foo', '//java:baz'],",
        ")");
    scratch.file("java/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = ['//a:bar'],",
        ")",
        "android_library(",
        "    name = 'lib',",
        "    deps = ['//a:bar'],",
        ")",
        "java_import(",
        "    name = 'baz',",
        "    jars = ['baz.jar'],",
        "    constraints = ['android'],",
        ")");
  }

  @Test
  public void testResourcesProvided() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:foo");

    NestedSet<ResourceContainer> directResources = aarImportTarget
        .getProvider(AndroidResourcesProvider.class)
        .getDirectAndroidResources();
    assertThat(directResources).hasSize(1);

    ResourceContainer resourceContainer = directResources.iterator().next();
    assertThat(resourceContainer.getManifest()).isNotNull();

    Iterable<Artifact> resourceArtifacts = resourceContainer.getArtifacts();
    assertThat(resourceArtifacts).hasSize(1);

    Artifact resourceTreeArtifact = resourceArtifacts.iterator().next();
    assertThat(resourceTreeArtifact.isTreeArtifact()).isTrue();
    assertThat(resourceTreeArtifact.getExecPathString()).endsWith("_aar/unzipped/resources/foo");
  }

  @Test
  public void testNativeLibsProvided() throws Exception {
    ConfiguredTarget androidLibraryTarget = getConfiguredTarget("//java:lib");

    NestedSet<Artifact> nativeLibs =
        androidLibraryTarget.getProvider(NativeLibsZipsProvider.class).getAarNativeLibs();
    assertThat(nativeLibs).containsExactly(
        actionsTestUtil().getFirstArtifactEndingWith(nativeLibs, "foo/native_libs.zip"),
        actionsTestUtil().getFirstArtifactEndingWith(nativeLibs, "bar/native_libs.zip"));
  }

  @Test
  public void testNativeLibsZipMakesItIntoApk() throws Exception {
    scratch.file("java/com/google/android/hello/BUILD",
        "aar_import(",
        "    name = 'my_aar',",
        "    aar = 'my_aar.aar',",
        ")",
        "android_binary(",
        "    name = 'my_app',",
        "    srcs = ['HelloApp.java'],",
        "    deps = [':my_aar'],",
        "    manifest = 'AndroidManifest.xml',",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:my_app");
    SpawnAction apkBuilderAction = (SpawnAction) actionsTestUtil()
        .getActionForArtifactEndingWith(getFilesToBuild(binary), "my_app_unsigned.apk");
    assertThat(
            Iterables.find(
                apkBuilderAction.getArguments(),
                Predicates.containsPattern("_aar/my_aar/native_libs.zip$")))
        .isNotEmpty();
  }


  @Test
  public void testClassesJarProvided() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:foo");

    Iterable<OutputJar> outputJars =
        aarImportTarget.getProvider(JavaRuleOutputJarsProvider.class).getOutputJars();
    assertThat(outputJars).hasSize(1);

    Artifact classesJar = outputJars.iterator().next().getClassJar();
    assertThat(classesJar.getFilename()).isEqualTo("classes_and_libs_merged.jar");

    SpawnAction jarMergingAction = ((SpawnAction) getGeneratingAction(classesJar));
    assertThat(jarMergingAction.getArguments()).contains("--dont_change_compression");
  }

  @Test
  public void testNoCustomJavaPackage() throws Exception {
    ResourceContainer resourceContainer = getConfiguredTarget("//a:foo")
        .getProvider(AndroidResourcesProvider.class)
        .getDirectAndroidResources()
        .iterator()
        .next();

    // aar_import should not set a custom java package. Instead aapt will read the
    // java package from the manifest.
    assertThat(resourceContainer.getJavaPackage()).isNull();
  }

  @Test
  public void testExportsPropagatesMergedAarJars() throws Exception {
    FileConfiguredTarget appTarget = getFileConfiguredTarget("//java:app.apk");
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(appTarget.getArtifact());

    ConfiguredTarget bar = getConfiguredTarget("//a:bar");
    Artifact barClassesJar =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "bar/classes_and_libs_merged.jar");
    // Verify that bar/classes_and_libs_merged.jar was in the artifact closure.
    assertThat(barClassesJar).isNotNull();
    assertThat(barClassesJar.getArtifactOwner().getLabel()).isEqualTo(bar.getLabel());

    ConfiguredTarget foo = getConfiguredTarget("//a:foo");
    Artifact fooClassesJar =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "foo/classes_and_libs_merged.jar");
    // Verify that foo/classes_and_libs_merged.jar was in the artifact closure.
    assertThat(fooClassesJar).isNotNull();
    assertThat(fooClassesJar.getArtifactOwner().getLabel()).isEqualTo(foo.getLabel());

    ConfiguredTarget baz = getConfiguredTarget("//java:baz.jar");
    Artifact bazJar =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "baz.jar");
    // Verify that baz.jar was in the artifact closure
    assertThat(bazJar).isNotNull();
    assertThat(bazJar.getArtifactOwner().getLabel()).isEqualTo(baz.getLabel());
  }

  @Test
  public void testExportsPropagatesResources() throws Exception {
    FileConfiguredTarget appTarget = getFileConfiguredTarget("//java:app.apk");
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(appTarget.getArtifact());

    ConfiguredTarget bar = getConfiguredTarget("//a:bar");
    Artifact barResources =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "_aar/unzipped/resources/bar");
    assertThat(barResources).isNotNull();
    assertThat(barResources.getArtifactOwner().getLabel()).isEqualTo(bar.getLabel());

    ConfiguredTarget foo = getConfiguredTarget("//a:foo");
    Artifact fooResources =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "_aar/unzipped/resources/foo");
    assertThat(fooResources).isNotNull();
    assertThat(fooResources.getArtifactOwner().getLabel()).isEqualTo(foo.getLabel());
  }

  @Test
  public void testJavaCompilationArgsProvider() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:bar");

    JavaCompilationArgsProvider provider = aarImportTarget
        .getProvider(JavaCompilationArgsProvider.class);
    assertThat(provider).isNotNull();
    assertThat(artifactsToStrings(provider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly(
            "bin a/_aar/bar/classes_and_libs_merged.jar",
            "bin a/_aar/foo/classes_and_libs_merged.jar",
            "src java/baz.jar");
  }
}
