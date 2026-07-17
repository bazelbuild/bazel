// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles.ConflictPolicy;
import com.google.devtools.build.lib.analysis.Runfiles.RunfilesConflictReceiver;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Runfiles}. */
@RunWith(JUnit4.class)
public final class RunfilesTest extends FoundationTestCase {

  private void checkWarning() {
    assertContainsEvent("obscured by a -> x");
    assertWithMessage("Runfiles.filterListForObscuringSymlinks should have warned once")
        .that(eventCollector.count())
        .isEqualTo(1);
    assertThat(Iterables.getOnlyElement(eventCollector).getKind()).isEqualTo(EventKind.WARNING);
  }

  private static StarlarkThread newStarlarkThread(String... options)
      throws OptionsParsingException {
    return StarlarkThread.createTransient(
        Mutability.create("test"),
        Options.parse(BuildLanguageOptions.class, options).getOptions().toStarlarkSemantics());
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurer() {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "x");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), ActionsTestUtil.createArtifact(root, "c/b"));
    assertThat(
            Runfiles.filterListForObscuringSymlinks(warningPrefixConflictReceiver(), obscuringMap))
        .containsExactly(pathA, artifactA);
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadGrandParentObscurer() {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "x");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b/c"), ActionsTestUtil.createArtifact(root, "b/c"));
    assertThat(
            Runfiles.filterListForObscuringSymlinks(warningPrefixConflictReceiver(), obscuringMap))
        .containsExactly(pathA, artifactA);
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurerNoListener() {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), ActionsTestUtil.createArtifact(root, "c/b"));
    assertThat(
            Runfiles.filterListForObscuringSymlinks(warningPrefixConflictReceiver(), obscuringMap))
        .containsExactly(pathA, artifactA);
  }

  @Test
  public void testFilterListForObscuringSymlinksIgnoresOkObscurer() {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), ActionsTestUtil.createArtifact(root, "a/b"));

    assertThat(
            Runfiles.filterListForObscuringSymlinks(warningPrefixConflictReceiver(), obscuringMap))
        .containsExactly(pathA, artifactA);
    assertNoEvents();
  }

  @Test
  public void testFilterListForObscuringSymlinksNoObscurers() {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a");
    obscuringMap.put(pathA, artifactA);
    PathFragment pathBC = PathFragment.create("b/c");
    Artifact artifactBC = ActionsTestUtil.createArtifact(root, "a/b");
    obscuringMap.put(pathBC, artifactBC);
    assertThat(
            Runfiles.filterListForObscuringSymlinks(warningPrefixConflictReceiver(), obscuringMap))
        .containsExactly(pathA, artifactA, pathBC, artifactBC);
    assertNoEvents();
  }

  private RunfilesConflictReceiver warningPrefixConflictReceiver() {
    return new RunfilesConflictReceiver() {
      @Override
      public void nestedRunfilesTree(Artifact runfilesTree) {
        throw new AssertionError(runfilesTree);
      }

      @Override
      public void prefixConflict(String message) {
        reporter.handle(Event.warn(message));
      }
    };
  }

  @Test
  public void testBuilderMergeConflictPolicyDefault() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build();
    Runfiles r2 = new Runfiles.Builder("TESTING").merge(r1).build();
    assertThat(r2.getConflictPolicy()).isEqualTo(ConflictPolicy.WARN);
  }

  @Test
  public void testBuilderMergeConflictPolicyInherit() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build().setConflictPolicy(ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING").merge(r1).build();
    assertThat(r2.getConflictPolicy()).isEqualTo(ConflictPolicy.WARN);
  }

  @Test
  public void testBuilderMergeConflictPolicyInheritStrictest() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build().setConflictPolicy(ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING").build().setConflictPolicy(ConflictPolicy.ERROR);
    Runfiles r3 = new Runfiles.Builder("TESTING").merge(r1).merge(r2).build();
    assertThat(r3.getConflictPolicy()).isEqualTo(ConflictPolicy.ERROR);
    // Swap ordering
    Runfiles r4 = new Runfiles.Builder("TESTING").merge(r2).merge(r1).build();
    assertThat(r4.getConflictPolicy()).isEqualTo(ConflictPolicy.ERROR);
  }

  @Test
  public void testRunfileAdded() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathB = PathFragment.create("repo/b");
    PathFragment legacyPathB = LabelConstants.EXTERNAL_PATH_PREFIX.getRelative(pathB);
    PathFragment runfilesPathB = LabelConstants.EXTERNAL_RUNFILES_PATH_PREFIX.getRelative(pathB);
    Artifact artifactB = ActionsTestUtil.createArtifactWithRootRelativePath(root, legacyPathB);

    Runfiles runfiles = new Runfiles.Builder("wsname").addSymlink(runfilesPathB, artifactB).build();

    assertThat(
            runfiles.getRunfilesInputs(
                warningPrefixConflictReceiver(), /* repoMappingManifest= */ null))
        .containsExactly(
            PathFragment.create("wsname/.runfile"), null, PathFragment.create("repo/b"), artifactB);
    assertNoEvents();
  }

  @Test
  public void testMergeWithSymlinks() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b/target");
    Runfiles runfilesA =
        new Runfiles.Builder("TESTING")
            .addSymlink(PathFragment.create("a/symlink"), artifactA)
            .build();
    Runfiles runfilesB =
        new Runfiles.Builder("TESTING")
            .addSymlink(PathFragment.create("b/symlink"), artifactB)
            .build();
    StarlarkThread thread = newStarlarkThread();

    Runfiles runfilesC = runfilesA.merge(runfilesB, thread);
    assertThat(runfilesC.getRunfilesInputs(/* repoMappingManifest= */ null))
        .containsExactly(
            PathFragment.create("TESTING/a/symlink"),
            artifactA,
            PathFragment.create("TESTING/b/symlink"),
            artifactB);
  }

  @Test
  public void mergeAll_symlinks() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b/target");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c/target");
    Runfiles runfilesA =
        new Runfiles.Builder("TESTING")
            .addSymlink(PathFragment.create("a/symlink"), artifactA)
            .build();
    Runfiles runfilesB =
        new Runfiles.Builder("TESTING")
            .addSymlink(PathFragment.create("b/symlink"), artifactB)
            .build();
    Runfiles runfilesC =
        new Runfiles.Builder("TESTING")
            .addSymlink(PathFragment.create("c/symlink"), artifactC)
            .build();
    StarlarkThread thread = newStarlarkThread();

    Runfiles runfilesMerged =
        runfilesA.mergeAll(StarlarkList.immutableOf(runfilesB, runfilesC), thread);
    assertThat(runfilesMerged.getRunfilesInputs(/* repoMappingManifest= */ null))
        .containsExactly(
            PathFragment.create("TESTING/a/symlink"),
            artifactA,
            PathFragment.create("TESTING/b/symlink"),
            artifactB,
            PathFragment.create("TESTING/c/symlink"),
            artifactC);
  }

  @Test
  public void testMergeEmptyWithNonEmpty() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Runfiles runfilesB = new Runfiles.Builder("TESTING").addArtifact(artifactA).build();
    StarlarkThread thread = newStarlarkThread();

    assertThat(Runfiles.EMPTY.merge(runfilesB, thread)).isSameInstanceAs(runfilesB);
    assertThat(runfilesB.merge(Runfiles.EMPTY, thread)).isSameInstanceAs(runfilesB);
  }

  @Test
  public void mergeAll_emptyWithNonEmpty() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifact = ActionsTestUtil.createArtifact(root, "target");
    Runfiles nonEmpty = new Runfiles.Builder("TESTING").addArtifact(artifact).build();
    StarlarkThread thread = newStarlarkThread();

    assertThat(Runfiles.EMPTY.mergeAll(StarlarkList.immutableOf(nonEmpty), thread))
        .isSameInstanceAs(nonEmpty);
    assertThat(
            Runfiles.EMPTY.mergeAll(
                StarlarkList.immutableOf(Runfiles.EMPTY, nonEmpty, Runfiles.EMPTY), thread))
        .isSameInstanceAs(nonEmpty);
    assertThat(nonEmpty.mergeAll(StarlarkList.immutableOf(Runfiles.EMPTY, Runfiles.EMPTY), thread))
        .isSameInstanceAs(nonEmpty);
    assertThat(nonEmpty.mergeAll(StarlarkList.immutableOf(), thread)).isSameInstanceAs(nonEmpty);
  }

  @Test
  public void mergeAll_emptyWithEmpty() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    assertThat(Runfiles.EMPTY.mergeAll(StarlarkList.immutableOf(), thread))
        .isSameInstanceAs(Runfiles.EMPTY);
    assertThat(
            Runfiles.EMPTY.mergeAll(
                StarlarkList.immutableOf(Runfiles.EMPTY, Runfiles.EMPTY), thread))
        .isSameInstanceAs(Runfiles.EMPTY);
  }

  @Test
  public void merge_exceedsDepthLimit_throwsException() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b/target");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c/target");
    Runfiles runfilesA = new Runfiles.Builder("TESTING").addArtifact(artifactA).build();
    Runfiles runfilesB = new Runfiles.Builder("TESTING").addArtifact(artifactB).build();
    Runfiles runfilesC = new Runfiles.Builder("TESTING").addArtifact(artifactC).build();
    StarlarkThread thread = newStarlarkThread("--nested_set_depth_limit=2");

    Runfiles mergeAB = runfilesA.merge(runfilesB, thread);
    EvalException expected =
        assertThrows(EvalException.class, () -> mergeAB.merge(runfilesC, thread));
    assertThat(expected).hasMessageThat().contains("artifacts depset depth 3 exceeds limit (2)");
  }

  @Test
  public void mergeAll_exceedsDepthLimit_throwsException() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b/target");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c/target");
    PathFragment sympathA = PathFragment.create("a/symlink");
    PathFragment sympathB = PathFragment.create("b/symlink");
    PathFragment sympathC = PathFragment.create("c/symlink");
    Runfiles runfilesA = new Runfiles.Builder("TESTING").addSymlink(sympathA, artifactA).build();
    Runfiles runfilesB = new Runfiles.Builder("TESTING").addSymlink(sympathB, artifactB).build();
    Runfiles runfilesC = new Runfiles.Builder("TESTING").addSymlink(sympathC, artifactC).build();
    StarlarkThread thread = newStarlarkThread("--nested_set_depth_limit=2");

    Runfiles mergeAllAB = runfilesA.mergeAll(StarlarkList.immutableOf(runfilesB), thread);
    EvalException expected =
        assertThrows(
            EvalException.class,
            () -> mergeAllAB.mergeAll(StarlarkList.immutableOf(runfilesC), thread));
    assertThat(expected).hasMessageThat().contains("symlinks depset depth 3 exceeds limit (2)");
  }

  @Test
  public void testGetEmptyFilenames() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifact = ActionsTestUtil.createArtifact(root, "my-artifact");
    Runfiles runfiles =
        new Runfiles.Builder("TESTING")
            .addArtifact(artifact)
            .addSymlink(PathFragment.create("my-symlink"), artifact)
            .addRootSymlink(PathFragment.create("my-root-symlink"), artifact)
            .setEmptyFilesSupplier(
                new Runfiles.EmptyFilesSupplier() {
                  @Override
                  public ImmutableList<PathFragment> getExtraPaths(
                      Set<PathFragment> manifestPaths) {
                    return manifestPaths.stream()
                        .map((f) -> f.replaceName(f.getBaseName() + "-empty"))
                        .collect(ImmutableList.toImmutableList());
                  }

                  @Override
                  public void fingerprint(Fingerprint fingerprint) {}
                })
            .build();
    assertThat(runfiles.getEmptyFilenames())
        .containsExactly(
            PathFragment.create("my-artifact-empty"), PathFragment.create("my-symlink-empty"));
  }
}
