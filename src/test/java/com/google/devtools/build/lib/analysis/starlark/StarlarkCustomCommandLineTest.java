// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.MissingExpansionException;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.HasDigest;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.starlark.StarlarkCustomCommandLine.VectorArg;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collection;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.Tuple;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkCustomCommandLine} */
@RunWith(JUnit4.class)
public class StarlarkCustomCommandLineTest {

  private static final ArtifactExpander EMPTY_EXPANDER = (artifact, output) -> {};

  private final Scratch scratch = new Scratch();
  private Path execRoot;
  private ArtifactRoot derivedRoot;

  @Before
  public void createArtifactRoot() throws IOException {
    execRoot = scratch.dir("execroot");
    derivedRoot = ArtifactRoot.asDerivedRoot(execRoot, false, "bin");
  }

  @Test
  public void vectorArgAddToFingerprint_treeArtifactMissingExpansion_returnsDigest()
      throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    StarlarkCustomCommandLine commandLine =
        createCustomCommandLine(new VectorArg.Builder(Tuple.of(tree)).setExpandDirectories(true));
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Fingerprint fingerprint = new Fingerprint();

    // TODO(b/167696101): Fail arguments computation when we are missing the directory from inputs.
    commandLine.addToFingerprint(actionKeyContext, EMPTY_EXPANDER, fingerprint);

    assertThat(fingerprint.digestAndReset()).isNotEmpty();
  }

  @Test
  public void vectorArgAddToFingerprint_expandFileset_includesInDigest() throws Exception {
    SpecialArtifact fileset = createFileset("fileset");
    StarlarkCustomCommandLine commandLine =
        createCustomCommandLine(
            new VectorArg.Builder(Tuple.of(fileset)).setExpandDirectories(true));
    FilesetOutputSymlink symlink1 = createFilesetSymlink("file1");
    FilesetOutputSymlink symlink2 = createFilesetSymlink("file2");
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Fingerprint fingerprint = new Fingerprint();
    ArtifactExpander artifactExpander =
        createArtifactExpander(
            /*treeExpansions=*/ ImmutableMap.of(),
            ImmutableMap.of(fileset, ImmutableList.of(symlink1, symlink2)));

    commandLine.addToFingerprint(actionKeyContext, artifactExpander, fingerprint);

    assertThat(fingerprint.digestAndReset()).isNotEmpty();
  }

  @Test
  public void vectorArgAddToFingerprint_expandTreeArtifact_includesInDigest() throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    StarlarkCustomCommandLine commandLine =
        createCustomCommandLine(new VectorArg.Builder(Tuple.of(tree)).setExpandDirectories(true));
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "child");
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Fingerprint fingerprint = new Fingerprint();
    ArtifactExpander artifactExpander =
        createArtifactExpander(
            ImmutableMap.of(tree, ImmutableList.of(child)),
            /*filesetExpansions*/ ImmutableMap.of());

    commandLine.addToFingerprint(actionKeyContext, artifactExpander, fingerprint);

    assertThat(fingerprint.digestAndReset()).isNotEmpty();
  }

  @Test
  public void vectorArgAddToFingerprint_expandFilesetMissingExpansion_fails() {
    SpecialArtifact fileset = createFileset("fileset");
    StarlarkCustomCommandLine commandLine =
        createCustomCommandLine(
            new VectorArg.Builder(Tuple.of(fileset)).setExpandDirectories(true));
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Fingerprint fingerprint = new Fingerprint();

    assertThrows(
        CommandLineExpansionException.class,
        () -> commandLine.addToFingerprint(actionKeyContext, EMPTY_EXPANDER, fingerprint));
  }

  @Test
  public void vectorArgArguments_expandsTreeArtifact() throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    StarlarkCustomCommandLine commandLine =
        createCustomCommandLine(new VectorArg.Builder(Tuple.of(tree)).setExpandDirectories(true));
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(tree, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(tree, "child2");
    ArtifactExpander artifactExpander =
        createArtifactExpander(
            ImmutableMap.of(tree, ImmutableList.of(child1, child2)),
            /*filesetExpansions*/ ImmutableMap.of());

    Iterable<String> arguments = commandLine.arguments(artifactExpander);

    assertThat(arguments).containsExactly("bin/tree/child1", "bin/tree/child2");
  }

  @Test
  public void vectorArgArguments_expandsFileset() throws Exception {
    SpecialArtifact fileset = createFileset("fileset");
    StarlarkCustomCommandLine commandLine =
        createCustomCommandLine(
            new VectorArg.Builder(Tuple.of(fileset)).setExpandDirectories(true));
    FilesetOutputSymlink symlink1 = createFilesetSymlink("file1");
    FilesetOutputSymlink symlink2 = createFilesetSymlink("file2");
    ArtifactExpander artifactExpander =
        createArtifactExpander(
            /*treeExpansions=*/ ImmutableMap.of(),
            ImmutableMap.of(fileset, ImmutableList.of(symlink1, symlink2)));

    Iterable<String> arguments = commandLine.arguments(artifactExpander);

    assertThat(arguments).containsExactly("bin/fileset/file1", "bin/fileset/file2");
  }

  @Test
  public void vectorArgArguments_treeArtifactMissingExpansion_returnsEmptyList() throws Exception {
    SpecialArtifact tree = createTreeArtifact("tree");
    StarlarkCustomCommandLine customCommandLine =
        createCustomCommandLine(new VectorArg.Builder(Tuple.of(tree)).setExpandDirectories(true));

    // TODO(b/167696101): Fail arguments computation when we are missing the directory from inputs.
    assertThat(customCommandLine.arguments(EMPTY_EXPANDER)).isEmpty();
  }

  @Test
  public void vectorArgArguments_filesetMissingExpansion_fails() {
    SpecialArtifact fileset = createFileset("fileset");
    StarlarkCustomCommandLine commandLine =
        createCustomCommandLine(
            new VectorArg.Builder(Tuple.of(fileset)).setExpandDirectories(true));

    assertThrows(CommandLineExpansionException.class, () -> commandLine.arguments(EMPTY_EXPANDER));
  }

  private SpecialArtifact createFileset(String relativePath) {
    return createSpecialArtifact(relativePath, SpecialArtifactType.FILESET);
  }

  private FilesetOutputSymlink createFilesetSymlink(String relativePath) {
    return FilesetOutputSymlink.create(
        PathFragment.create(relativePath),
        PathFragment.EMPTY_FRAGMENT,
        mock(HasDigest.class),
        /*isGeneratedTarget=*/ false,
        execRoot.asFragment());
  }

  private SpecialArtifact createTreeArtifact(String relativePath) {
    SpecialArtifact tree = createSpecialArtifact(relativePath, SpecialArtifactType.TREE);
    tree.setGeneratingActionKey(ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0));
    return tree;
  }

  private SpecialArtifact createSpecialArtifact(String relativePath, SpecialArtifactType type) {
    return new SpecialArtifact(
        derivedRoot,
        derivedRoot.getExecPath().getRelative(relativePath),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        type);
  }

  private static StarlarkCustomCommandLine createCustomCommandLine(
      VectorArg.Builder vectorArgBuilder) {
    return new StarlarkCustomCommandLine.Builder(StarlarkSemantics.DEFAULT)
        .add(vectorArgBuilder)
        .build(/*flagPerLine=*/ false);
  }

  private static ArtifactExpander createArtifactExpander(
      ImmutableMap<SpecialArtifact, ImmutableList<TreeFileArtifact>> treeExpansions,
      ImmutableMap<SpecialArtifact, ImmutableList<FilesetOutputSymlink>> filesetExpansions) {
    return new ArtifactExpander() {
      @Override
      public void expand(Artifact artifact, Collection<? super Artifact> output) {
        //noinspection SuspiciousMethodCalls
        ImmutableList<TreeFileArtifact> expansion = treeExpansions.get(artifact);
        if (expansion != null) {
          output.addAll(expansion);
        }
      }

      @Override
      public ImmutableList<FilesetOutputSymlink> getFileset(Artifact artifact)
          throws MissingExpansionException {
        //noinspection SuspiciousMethodCalls
        ImmutableList<FilesetOutputSymlink> filesetLinks = filesetExpansions.get(artifact);
        if (filesetLinks == null) {
          throw new MissingExpansionException("Cannot expand " + artifact);
        }
        return filesetLinks;
      }
    };
  }
}
