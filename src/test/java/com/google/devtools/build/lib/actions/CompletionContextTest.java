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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CompletionContext.ArtifactReceiver;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link CompletionContextTest}. */
@RunWith(JUnit4.class)
public class CompletionContextTest {
  private static final FileArtifactValue DUMMY_METADATA =
      new RemoteFileArtifactValue(/*digest=*/ new byte[0], /*size=*/ 0, /*locationIndex=*/ 0);

  private Path execRoot;
  private ArtifactRoot outputRoot;

  @Before
  public void createRoots() throws IOException {
    execRoot = new Scratch().dir("/execroot");
    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "out");
  }

  @Test
  public void visitArtifacts_expandsTreeArtifactWithPresentExpansion() {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact treeFile1 = TreeFileArtifact.createTreeOutput(tree, "file1");
    TreeFileArtifact treeFile2 = TreeFileArtifact.createTreeOutput(tree, "file2");
    ActionInputMap inputMap = new ActionInputMap(0);
    inputMap.putWithNoDepOwner(tree, DUMMY_METADATA);
    inputMap.putWithNoDepOwner(treeFile1, DUMMY_METADATA);
    inputMap.putWithNoDepOwner(treeFile2, DUMMY_METADATA);
    CompletionContext completionContext =
        createNewCompletionContext(
            ImmutableMap.of(tree, ImmutableList.of(treeFile1, treeFile2)), inputMap);

    List<Artifact> visited = new ArrayList<>();
    completionContext.visitArtifacts(ImmutableList.of(tree), createReceiver(visited));

    assertThat(visited).containsExactly(treeFile1, treeFile2).inOrder();
  }

  @Test
  public void visitArtifacts_skipsOmittedTreeArtifact() {
    SpecialArtifact tree = createTreeArtifact("tree");
    ActionInputMap inputMap = new ActionInputMap(0);
    inputMap.putWithNoDepOwner(tree, FileArtifactValue.OMITTED_FILE_MARKER);
    CompletionContext completionContext =
        createNewCompletionContext(/*expandedArtifacts=*/ ImmutableMap.of(), inputMap);

    List<Artifact> visited = new ArrayList<>();
    completionContext.visitArtifacts(ImmutableList.of(tree), createReceiver(visited));

    assertThat(visited).isEmpty();
  }

  private static ArtifactReceiver createReceiver(List<Artifact> visitedList) {
    return new ArtifactReceiver() {
      @Override
      public void accept(Artifact artifact) {
        visitedList.add(artifact);
      }

      @Override
      public void acceptFilesetMapping(Artifact fileset, PathFragment relName, Path targetFile) {}
    };
  }

  private SpecialArtifact createTreeArtifact(String relativeExecRootPath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        outputRoot, outputRoot.getExecPath().getRelative(relativeExecRootPath));
  }

  private CompletionContext createNewCompletionContext(
      ImmutableMap<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
      ActionInputMap inputMap) {
    return new CompletionContext(
        execRoot,
        expandedArtifacts,
        /*expandedFilesets=*/ ImmutableMap.of(),
        ArtifactPathResolver.IDENTITY,
        inputMap,
        /*expandFilesets=*/ false,
        /*fullyResolveFilesetLinks=*/ false);
  }
}
