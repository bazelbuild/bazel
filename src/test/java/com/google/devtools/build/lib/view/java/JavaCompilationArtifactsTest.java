// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.view.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit Test for {@link JavaCompilationArtifacts} */
@RunWith(JUnit4.class)
public class JavaCompilationArtifactsTest extends FoundationTestCase {

  @Test
  public void testSimple() {
    Path execRoot = scratch.getFileSystem().getPath("/exec");
    String outSegment = "root";
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, outSegment);
    Artifact runtimeJar = ActionsTestUtil.createArtifact(root, "rt.jar");
    Artifact compileTimeJar = ActionsTestUtil.createArtifact(root, "ct.jar");
    JavaCompilationArtifacts original =
        new JavaCompilationArtifacts.Builder()
            .addRuntimeJar(runtimeJar)
            .addCompileTimeJarAsFullJar(compileTimeJar)
            .build();
    assertThat(original.getRuntimeJars()).containsExactly(runtimeJar);
    assertThat(original.getCompileTimeJars()).containsExactly(compileTimeJar);
  }

  @Test
  public void testDuplicates() {
    Path execRoot = scratch.getFileSystem().getPath("/exec");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "root");
    Artifact runtimeJar = ActionsTestUtil.createArtifact(root, "rt.jar");
    Artifact compileTimeJar = ActionsTestUtil.createArtifact(root, "ct.jar");

    JavaCompilationArtifacts.Builder builder = new JavaCompilationArtifacts.Builder();
    // Add the same jars to the builder twice.
    builder.addRuntimeJar(runtimeJar).addCompileTimeJarAsFullJar(compileTimeJar);
    builder
        .addRuntimeJar(runtimeJar)
        .addCompileTimeJarAsFullJar(compileTimeJar)
        .build();
    JavaCompilationArtifacts artifacts = builder.build();
    // There is only a single instance of each jar in the final artifact collection.
    assertThat(artifacts.getRuntimeJars()).containsExactly(runtimeJar);
    assertThat(artifacts.getCompileTimeJars()).containsExactly(compileTimeJar);
  }
}
