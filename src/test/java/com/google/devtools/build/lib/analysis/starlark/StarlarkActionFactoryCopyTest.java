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
package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import net.starlark.java.eval.EvalException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests the argument validation of {@code ctx.actions.copy}.
 *
 * <p>Fileset artifacts cannot be created from Starlark in Bazel, so this is a unit test.
 */
@RunWith(JUnit4.class)
public class StarlarkActionFactoryCopyTest {
  private ArtifactRoot root;

  @Before
  public void setUp() {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path execRoot = fs.getPath("/exec");
    root = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
  }

  @Test
  public void fileToFile_andTreeToTree_areAccepted() throws Exception {
    StarlarkActionFactory.checkCopyArtifacts(
        ActionsTestUtil.createArtifact(root, "out"), ActionsTestUtil.createArtifact(root, "in"));
    StarlarkActionFactory.checkCopyArtifacts(
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(root, "out_tree"),
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(root, "in_tree"));
  }

  @Test
  public void filesetSource_isRejected() {
    SpecialArtifact fileset = ActionsTestUtil.createFilesetArtifact(root, "fs");
    Artifact out = ActionsTestUtil.createArtifact(root, "out");

    EvalException e =
        assertThrows(
            EvalException.class, () -> StarlarkActionFactory.checkCopyArtifacts(out, fileset));

    assertThat(e).hasMessageThat().contains("copy() of Fileset artifacts is not supported");
  }

  @Test
  public void filesetOutput_isRejected() {
    SpecialArtifact fileset = ActionsTestUtil.createFilesetArtifact(root, "fs");
    Artifact in = ActionsTestUtil.createArtifact(root, "in");

    EvalException e =
        assertThrows(
            EvalException.class, () -> StarlarkActionFactory.checkCopyArtifacts(fileset, in));

    assertThat(e).hasMessageThat().contains("copy() of Fileset artifacts is not supported");
  }

  @Test
  public void filesetToTree_isRejectedAsFilesetNotAsTypeMismatch() {
    SpecialArtifact fileset = ActionsTestUtil.createFilesetArtifact(root, "fs");
    SpecialArtifact tree = ActionsTestUtil.createTreeArtifactWithGeneratingAction(root, "tree");

    EvalException e =
        assertThrows(
            EvalException.class, () -> StarlarkActionFactory.checkCopyArtifacts(tree, fileset));

    assertThat(e).hasMessageThat().contains("Fileset");
  }
}
