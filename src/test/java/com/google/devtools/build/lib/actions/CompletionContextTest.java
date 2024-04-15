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
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CompletionContext.ArtifactReceiver;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.InOrder;

/** Test for {@link CompletionContextTest}. */
@RunWith(JUnit4.class)
public final class CompletionContextTest {
  private static final FileArtifactValue DUMMY_METADATA =
      RemoteFileArtifactValue.create(
          /* digest= */ new byte[0],
          /* size= */ 0,
          /* locationIndex= */ 0,
          /* expireAtEpochMilli= */ -1);

  private final ActionInputMap inputMap = new ActionInputMap(BugReporter.defaultInstance(), 0);
  private final Map<Artifact, ImmutableSortedSet<TreeFileArtifact>> treeExpansions =
      new HashMap<>();
  private final Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetExpansions =
      new HashMap<>();
  private Path execRoot;
  private ArtifactRoot outputRoot;

  @Before
  public void createRoots() throws Exception {
    execRoot = new Scratch().dir("/execroot");
    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "out");
  }

  @Test
  public void regularArtifact() {
    Artifact file = ActionsTestUtil.createArtifact(outputRoot, "file");
    inputMap.put(file, DUMMY_METADATA, /* depOwner= */ null);
    CompletionContext ctx = createCompletionContext(/* expandFilesets= */ true);

    assertThat(visit(ctx, file)).containsExactly(file);
    assertThat(expand(ctx, file)).containsExactly(file);
    assertThat(owners(ctx, file)).isEmpty();
  }

  @Test
  public void treeArtifact_present() {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact treeFile1 = TreeFileArtifact.createTreeOutput(tree, "file1");
    TreeFileArtifact treeFile2 = TreeFileArtifact.createTreeOutput(tree, "file2");
    TreeArtifactValue treeValue =
        TreeArtifactValue.newBuilder(tree)
            .putChild(treeFile1, DUMMY_METADATA)
            .putChild(treeFile2, DUMMY_METADATA)
            .build();
    inputMap.putTreeArtifact(tree, treeValue, /* depOwner= */ null);
    treeExpansions.put(tree, treeValue.getChildren());
    CompletionContext ctx = createCompletionContext(/* expandFilesets= */ true);

    assertThat(visit(ctx, tree)).containsExactly(treeFile1, treeFile2).inOrder();
    assertThat(expand(ctx, tree)).containsExactly(treeFile1, treeFile2).inOrder();
    assertThat(owners(ctx, treeFile1)).containsExactly(tree);
    assertThat(owners(ctx, treeFile2)).containsExactly(tree);
  }

  @Test
  public void treeArtifact_omitted() {
    SpecialArtifact tree = createTreeArtifact("tree");
    inputMap.putTreeArtifact(tree, TreeArtifactValue.OMITTED_TREE_MARKER, /* depOwner= */ null);
    CompletionContext ctx = createCompletionContext(/* expandFilesets= */ true);

    assertThat(visit(ctx, tree)).isEmpty();
    assertThat(expand(ctx, tree)).isEmpty();
  }

  @Test
  public void fileset_noExpansion() {
    SpecialArtifact fileset = createFileset("fs");
    inputMap.put(fileset, DUMMY_METADATA, /* depOwner= */ null);
    filesetExpansions.put(
        fileset, ImmutableList.of(filesetLink("a1", "b1"), filesetLink("a2", "b2")));
    CompletionContext ctx = createCompletionContext(/* expandFilesets= */ false);

    ArtifactReceiver receiver = mock(ArtifactReceiver.class);
    ctx.visitArtifacts(ImmutableList.of(fileset), receiver);
    verifyNoInteractions(receiver);

    assertThat(expand(ctx, fileset)).isEmpty();
  }

  @Test
  public void fileset_withExpansion() {
    SpecialArtifact fileset = createFileset("fs");
    inputMap.put(fileset, DUMMY_METADATA, /* depOwner= */ null);
    filesetExpansions.put(
        fileset, ImmutableList.of(filesetLink("a1", "b1"), filesetLink("a2", "b2")));
    CompletionContext ctx = createCompletionContext(/* expandFilesets= */ true);

    ArtifactReceiver receiver = mock(ArtifactReceiver.class);
    ctx.visitArtifacts(ImmutableList.of(fileset), receiver);
    InOrder inOrder = inOrder(receiver);
    inOrder
        .verify(receiver)
        .acceptFilesetMapping(fileset, PathFragment.create("a1"), execRoot.getRelative("b1"));
    inOrder
        .verify(receiver)
        .acceptFilesetMapping(fileset, PathFragment.create("a2"), execRoot.getRelative("b2"));

    ActionInput input1 = ActionInputHelper.fromPath(execRoot.getRelative("b1").asFragment());
    ActionInput input2 = ActionInputHelper.fromPath(execRoot.getRelative("b2").asFragment());
    assertThat(expand(ctx, fileset)).containsExactly(input1, input2).inOrder();
    assertThat(owners(ctx, input1)).containsExactly(fileset);
    assertThat(owners(ctx, input2)).containsExactly(fileset);
  }

  private static List<Artifact> visit(CompletionContext ctx, Artifact artifact) {
    List<Artifact> visited = new ArrayList<>();
    ctx.visitArtifacts(
        ImmutableList.of(artifact),
        new ArtifactReceiver() {
          @Override
          public void accept(Artifact artifact) {
            visited.add(artifact);
          }

          @Override
          public void acceptFilesetMapping(
              Artifact fileset, PathFragment relName, Path targetFile) {
            throw new AssertionError(fileset);
          }
        });
    return visited;
  }

  private static ImmutableList<? extends ActionInput> expand(
      CompletionContext ctx, Artifact artifact) {
    return ctx.expand(ImmutableList.of(artifact));
  }

  private static ImmutableSet<Artifact> owners(CompletionContext ctx, ActionInput input) {
    return ctx.getDepOwners(ImmutableList.of(input)).getDepOwners(input);
  }

  private SpecialArtifact createTreeArtifact(String rootRelativePath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        outputRoot, outputRoot.getExecPath().getRelative(rootRelativePath));
  }

  private SpecialArtifact createFileset(String rootRelativePath) {
    return SpecialArtifact.create(
        outputRoot,
        outputRoot.getExecPath().getRelative(rootRelativePath),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        SpecialArtifactType.FILESET);
  }

  private FilesetOutputSymlink filesetLink(String from, String to) {
    return FilesetOutputSymlink.createForTesting(
        PathFragment.create(from), execRoot.getRelative(to).asFragment(), execRoot.asFragment());
  }

  private CompletionContext createCompletionContext(boolean expandFilesets) {
    return new CompletionContext(
        execRoot,
        ImmutableMap.copyOf(treeExpansions),
        ImmutableMap.copyOf(filesetExpansions),
        ArtifactPathResolver.IDENTITY,
        inputMap,
        expandFilesets,
        /* fullyResolveFilesetLinks= */ false);
  }
}
