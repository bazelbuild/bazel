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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ApkManifestAction}. */
@RunWith(JUnit4.class)
public class ApkManifestActionTest {

  private FileSystem fileSystem;

  @Before
  public void setup() {
    fileSystem = FileSystems.getJavaIoFileSystem();
  }

  /** A regression test to make sure the action's key changes when the output manifest changes. */
  @Test
  public void testActionKey() throws Exception {
    Artifact outputFile = createArtifact("/workspace/java/test/manifest");
    AndroidSdkProvider sdk =
        AndroidSdkProvider.create(
            "24.0.3",
            createArtifact("/workspace/androidsdk/frameworkAidl"),
            null,  // aidlLib, optional
            createArtifact("/workspace/androidsdk/androidJar"),
            createArtifact("/workspace/androidsdk/shrinkedAndroidJar"),
            createArtifact("/workspace/androidsdk/annotationsJar"),
            createArtifact("/workspace/androidsdk/mainDexClasses"),
            createFilesToRunProvider("adb"),
            createFilesToRunProvider("dx"),
            createFilesToRunProvider("mainDexListCreator"),
            createFilesToRunProvider("aidl"),
            createFilesToRunProvider("aapt"),
            /* apkBuilder = */ null,
            createFilesToRunProvider("apkSigner"),
            createFilesToRunProvider("proguard"),
            createFilesToRunProvider("zipalign"),
            createFilesToRunProvider("resourceExtractor"));

    Iterable<Artifact> jars1 = ImmutableList.of(
        createArtifact("/workspace/java/test/output_jar1"),
        createArtifact("/workspace/java/test/output_jar2"));

    Iterable<Artifact> jars2 = ImmutableList.of(
        createArtifact("/workspace/java/test/output_jar1"),
        createArtifact("/workspace/java/test/output_jar2"),
        createArtifact("/workspace/java/test/output_jar3"));

    ResourceApk resourceApk = new ResourceApk(
        createArtifact("/workspace/java/test/resources.ap_"), // resourceApk
        null, // resourceJavaSrcJar
        null, // resourceJavaClassJar
        null, // resourceDeps
        null, // primaryResources
        createArtifact("/workspace/java/test/merged_manifest.xml"), // manifest
        null, // resourceProguardConfig
        null, // mainDexProguardConfig
        false /* legacy */);

    NativeLibs nativeLibs = new NativeLibs(
        ImmutableMap.<String, Iterable<Artifact>>of(
            "x86", ImmutableList.of(createArtifact("/workspace/java/test/x86.so")),
            "arm", ImmutableList.of(createArtifact("/workspace/java/test/arm.so"))),
        null /* nativeLibsName */);

    Artifact debugKeystore = createArtifact("/workspace/tools/android/debug_keystore");

    ApkManifestAction action1 = new ApkManifestAction(
        ActionsTestUtil.NULL_ACTION_OWNER,
        outputFile,
        true, /* textOutput */
        sdk,
        jars1,
        resourceApk,
        nativeLibs,
        debugKeystore);

    ApkManifestAction action2 = new ApkManifestAction(
        ActionsTestUtil.NULL_ACTION_OWNER,
        outputFile,
        true, /* textOutput */
        sdk,
        jars2,
        resourceApk,
        nativeLibs,
        debugKeystore);

    String key1 = action1.computeKey();
    String key2 = action2.computeKey();
    // Action 2 has 1 more jar than Action 1, so their manifests should be different, and therefore
    // their keys should also be different.
    assertThat(key1).isNotEqualTo(key2);
  }

  private Artifact createArtifact(String path) {
    Path p = fileSystem.getPath(path);
    Root root = Root.asSourceRoot(fileSystem.getRootDirectory());
    try {
      return new Artifact(
          p,
          root,
          root.getExecPath().getRelative(p.relativeTo(root.getPath())),
          new LabelArtifactOwner(Label.parseAbsolute("//foo:bar")));
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  private FilesToRunProvider createFilesToRunProvider(String name) {
    return new FilesToRunProvider(null, null, createArtifact("/workspace/androidsdk/" + name));
  }
}

