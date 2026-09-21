// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule;
import com.google.devtools.build.buildjar.javac.plugins.processing.AnnotationProcessingModule;
import java.nio.file.Path;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that the per-invocation worker-sandbox prefix is stripped from output paths, so that jdeps
 * and annotation-processing manifest outputs are deterministic across sandbox slots (see #29468).
 */
@RunWith(JUnit4.class)
public class JavacWorkDirStrippingTest {

  @Test
  public void stripWorkDir_stripsSandboxPrefix() {
    Path workDir = Path.of("/exec/__sandbox/751/_main");
    Path source = workDir.resolve("src/com/example/Foo.java");
    assertThat(DependencyModule.stripWorkDir(workDir, source))
        .isEqualTo("src/com/example/Foo.java");
  }

  @Test
  public void stripWorkDir_isDeterministicAcrossSlots() {
    Path slot1 = Path.of("/exec/__sandbox/751/_main");
    Path slot2 = Path.of("/exec/__sandbox/1088/_main");
    assertThat(DependencyModule.stripWorkDir(slot1, slot1.resolve("libfoo.jar")))
        .isEqualTo(DependencyModule.stripWorkDir(slot2, slot2.resolve("libfoo.jar")));
  }

  @Test
  public void stripWorkDir_leavesPathOutsideWorkDirUnchanged() {
    Path workDir = Path.of("/exec/__sandbox/751/_main");
    Path outside = Path.of("/other/location/libfoo.jar");
    assertThat(DependencyModule.stripWorkDir(workDir, outside)).isEqualTo(outside.toString());
  }

  @Test
  public void stripWorkDir_leavesPathUnchangedForEmptyWorkDir() {
    Path path = Path.of("src/com/example/Foo.java");
    assertThat(DependencyModule.stripWorkDir(Path.of(""), path)).isEqualTo(path.toString());
  }

  @Test
  public void stripSourceRoot_stripsWorkDirPrefix() {
    Path workDir = Path.of("/exec/__sandbox/751/_main");
    AnnotationProcessingModule module = module(workDir, workDir.resolve("_sourcegenfiles"));
    Path source = workDir.resolve("src/com/example/Foo.java");
    assertThat(module.stripSourceRoot(source).toString()).isEqualTo("src/com/example/Foo.java");
  }

  @Test
  public void stripSourceRoot_stripsSourceGenDirForGeneratedFiles() {
    Path workDir = Path.of("/exec/__sandbox/751/_main");
    Path sourceGenDir = workDir.resolve("_sourcegenfiles");
    AnnotationProcessingModule module = module(workDir, sourceGenDir);
    Path generated = sourceGenDir.resolve("com/example/Gen.java");
    assertThat(module.stripSourceRoot(generated).toString()).isEqualTo("com/example/Gen.java");
  }

  private static AnnotationProcessingModule module(Path workDir, Path sourceGenDir) {
    AnnotationProcessingModule.Builder builder = AnnotationProcessingModule.builder();
    builder.setSourceGenDir(sourceGenDir);
    builder.setManifestProtoPath(Path.of("/tmp/manifest.proto"));
    builder.setWorkDir(workDir);
    return builder.build();
  }
}
