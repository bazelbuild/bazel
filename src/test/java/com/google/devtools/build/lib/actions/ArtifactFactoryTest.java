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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ARTIFACT_OWNER;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ArtifactFactory}. Also see {@link ArtifactTest} for a test
 * of individual artifacts.
 */
@RunWith(JUnit4.class)
public class ArtifactFactoryTest {

  private static final RepositoryName MAIN = RepositoryName.MAIN;

  private Scratch scratch = new Scratch();

  private Path execRoot;
  private Root clientRoot;
  private Root clientRoRoot;
  private Root alienRoot;
  private ArtifactRoot outRoot;

  private PathFragment fooPath;
  private PackageIdentifier fooPackage;
  private PathFragment fooRelative;

  private PathFragment barPath;
  private PackageIdentifier barPackage;
  private PathFragment barRelative;

  private PathFragment alienPath;
  private PackageIdentifier alienPackage;
  private PathFragment alienRelative;

  private ArtifactFactory artifactFactory;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  @Before
  public final void createFiles() throws Exception  {
    execRoot = scratch.dir("/output/workspace");
    clientRoot = Root.fromPath(scratch.dir("/client/workspace"));
    clientRoRoot = Root.fromPath(scratch.dir("/client/RO/workspace"));
    alienRoot = Root.fromPath(scratch.dir("/client/workspace"));
    outRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out-root", "x", "bin");

    fooPath = PathFragment.create("foo");
    fooPackage = PackageIdentifier.createInMainRepo(fooPath);
    fooRelative = fooPath.getRelative("foosource.txt");

    barPath = PathFragment.create("foo/bar");
    barPackage = PackageIdentifier.createInMainRepo(barPath);
    barRelative = barPath.getRelative("barsource.txt");

    alienPath = PathFragment.create("external/alien");
    alienPackage = PackageIdentifier.create("alien", alienPath);
    alienRelative = alienPath.getRelative("alien.txt");

    artifactFactory = new ArtifactFactory(execRoot.getParentDirectory(), "bazel-out");
    setupRoots();
  }

  private void setupRoots() {
    Map<PackageIdentifier, Root> packageRootMap = new HashMap<>();
    packageRootMap.put(fooPackage, clientRoot);
    packageRootMap.put(barPackage, clientRoRoot);
    packageRootMap.put(alienPackage, alienRoot);
    artifactFactory.setPackageRoots(packageRootMap::get);
  }

  @Test
  public void testGetSourceArtifactYieldsSameArtifact() throws Exception {
    assertThat(artifactFactory.getSourceArtifact(fooRelative, clientRoot))
        .isSameInstanceAs(artifactFactory.getSourceArtifact(fooRelative, clientRoot));
  }

  @Test
  public void testGetSourceArtifactUnnormalized() throws Exception {
    assertThat(
            artifactFactory.getSourceArtifact(
                PathFragment.create("foo/./foosource.txt"), clientRoot))
        .isSameInstanceAs(artifactFactory.getSourceArtifact(fooRelative, clientRoot));
  }

  @Test
  public void testResolveArtifact_noDerived_simpleSource() throws Exception {
    assertThat(artifactFactory.resolveSourceArtifact(fooRelative, MAIN))
        .isSameInstanceAs(artifactFactory.getSourceArtifact(fooRelative, clientRoot));
    assertThat(artifactFactory.resolveSourceArtifact(barRelative, MAIN))
        .isSameInstanceAs(artifactFactory.getSourceArtifact(barRelative, clientRoRoot));
  }

  @Test
  public void testResolveArtifact_inExternalRepo() throws Exception {
    Artifact a1 = artifactFactory.getSourceArtifact(alienRelative, alienRoot);
    Artifact a2 = artifactFactory.resolveSourceArtifact(alienRelative, MAIN);
    assertThat(a1).isSameInstanceAs(a2);
  }

  @Test
  public void testResolveArtifact_noDerived_derivedRoot() throws Exception {
    assertThat(
            artifactFactory.resolveSourceArtifact(
                outRoot.getRoot().getRelative(fooRelative).relativeTo(execRoot), MAIN))
        .isNull();
    assertThat(
            artifactFactory.resolveSourceArtifact(
                outRoot.getRoot().getRelative(barRelative).relativeTo(execRoot), MAIN))
        .isNull();
  }

  @Test
  public void testResolveArtifact_noDerived_simpleSource_other() throws Exception {
    Artifact actual = artifactFactory.resolveSourceArtifact(fooRelative, MAIN);
    assertThat(actual).isSameInstanceAs(artifactFactory.getSourceArtifact(fooRelative, clientRoot));
    actual = artifactFactory.resolveSourceArtifact(barRelative, MAIN);
    assertThat(actual)
        .isSameInstanceAs(artifactFactory.getSourceArtifact(barRelative, clientRoRoot));
  }

  @Test
  public void testResolveArtifactWithUpLevelFailsCleanly() throws Exception {
    // We need a package in the root directory to make every exec path (even one with up-level
    // references) be in a package.
    Map<PackageIdentifier, Root> packageRoots =
        ImmutableMap.of(PackageIdentifier.createInMainRepo(PathFragment.create("")), clientRoot);
    artifactFactory.setPackageRoots(packageRoots::get);
    PathFragment outsideWorkspace = PathFragment.create("../foo");
    PathFragment insideWorkspace = PathFragment.create("../workspace/foo");
    assertThat(artifactFactory.resolveSourceArtifact(outsideWorkspace, MAIN)).isNull();
    assertWithMessage(
            "Up-level-containing paths that descend into the right workspace aren't allowed")
        .that(artifactFactory.resolveSourceArtifact(insideWorkspace, MAIN))
        .isNull();
    MockPackageRootResolver packageRootResolver = new MockPackageRootResolver();
    packageRootResolver.setPackageRoots(packageRoots);
    Map<PathFragment, Artifact> result = new HashMap<>();
    result.put(insideWorkspace, null);
    result.put(outsideWorkspace, null);
    assertThat(
        artifactFactory.resolveSourceArtifacts(ImmutableList.of(insideWorkspace, outsideWorkspace),
            packageRootResolver).entrySet()).containsExactlyElementsIn(result.entrySet());
  }

  @Test
  public void testClearResetsFactory() {
    Artifact fooArtifact = artifactFactory.getSourceArtifact(fooRelative, clientRoot);
    artifactFactory.clear();
    setupRoots();
    assertThat(artifactFactory.getSourceArtifact(fooRelative, clientRoot))
        .isNotSameInstanceAs(fooArtifact);
  }

  @Test
  public void testFindDerivedRoot() throws Exception {
    assertThat(artifactFactory.isDerivedArtifact(fooRelative)).isFalse();
    assertThat(artifactFactory.isDerivedArtifact(
        PathFragment.create("bazel-out/local-fastbuild/bin/foo"))).isTrue();
  }

  @Test
  public void testAbsoluteArtifact() throws Exception {
    Root absoluteRoot = Root.absoluteRoot(scratch.getFileSystem());

    assertThat(
            artifactFactory.getSourceArtifact(PathFragment.create("foo"), clientRoot).getExecPath())
        .isEqualTo(PathFragment.create("foo"));
    assertThat(
            artifactFactory
                .getSourceArtifact(PathFragment.create("/foo"), absoluteRoot)
                .getExecPath())
        .isEqualTo(PathFragment.create("/foo"));
    assertThrows(
        IllegalArgumentException.class,
        () -> artifactFactory.getSourceArtifact(PathFragment.create("/foo"), clientRoot));
    assertThrows(
        IllegalArgumentException.class,
        () -> artifactFactory.getSourceArtifact(PathFragment.create("foo"), absoluteRoot));
  }

  @Test
  public void testSetGeneratingActionIdempotenceNewActionGraph() throws Exception {
    Artifact.DerivedArtifact a =
        artifactFactory.getDerivedArtifact(fooRelative, outRoot, NULL_ARTIFACT_OWNER);
    Artifact.DerivedArtifact b =
        artifactFactory.getDerivedArtifact(barRelative, outRoot, NULL_ARTIFACT_OWNER);
    a.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    b.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    MutableActionGraph actionGraph = new MapBasedActionGraph(actionKeyContext);
    Action originalAction = new ActionsTestUtil.NullAction(NULL_ACTION_OWNER, a);
    actionGraph.registerAction(originalAction);

    // Creating a second Action referring to the Artifact should create a conflict.
    Action action = new ActionsTestUtil.NullAction(NULL_ACTION_OWNER, a, b);
    ActionConflictException e =
        assertThrows(ActionConflictException.class, () -> actionGraph.registerAction(action));
    assertThat(e.getArtifact()).isSameInstanceAs(a);
    assertThat(actionGraph.getGeneratingAction(a)).isSameInstanceAs(originalAction);
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_exactMatch() {
    artifactFactory.noteAnalysisStarting();
    Artifact.SourceArtifact original = artifactFactory.getSourceArtifact(fooRelative, clientRoot);

    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(fooRelative, MAIN);

    assertThat(result).containsExactly(original);
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_upperCaseLookupFindsLowerCaseArtifact() {
    artifactFactory.noteAnalysisStarting();
    PathFragment lowerPath = PathFragment.create("foo/foosource.txt");
    Artifact.SourceArtifact original = artifactFactory.getSourceArtifact(lowerPath, clientRoot);

    PathFragment upperPath = PathFragment.create("foo/FooSource.txt");
    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(upperPath, MAIN);

    assertThat(result).containsExactly(original);
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_lowerCaseLookupFindsUpperCaseArtifact() {
    artifactFactory.noteAnalysisStarting();
    PathFragment upperPath = PathFragment.create("foo/FooSource.txt");
    Artifact.SourceArtifact original = artifactFactory.getSourceArtifact(upperPath, clientRoot);

    PathFragment lowerPath = PathFragment.create("foo/foosource.txt");
    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(lowerPath, MAIN);

    assertThat(result).containsExactly(original);
  }

  @Test
  public void testGetSourceArtifactDifferentCasings_returnsDifferentArtifacts() {
    artifactFactory.noteAnalysisStarting();
    PathFragment lower = PathFragment.create("foo/header.h");
    PathFragment upper = PathFragment.create("foo/Header.h");
    Artifact.SourceArtifact lowerArtifact = artifactFactory.getSourceArtifact(lower, clientRoot);
    Artifact.SourceArtifact upperArtifact = artifactFactory.getSourceArtifact(upper, clientRoot);

    assertThat(upperArtifact).isNotSameInstanceAs(lowerArtifact);
    assertThat(lowerArtifact.getExecPath()).isEqualTo(lower);
    assertThat(upperArtifact.getExecPath()).isEqualTo(upper);
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_multipleMatches() {
    artifactFactory.noteAnalysisStarting();
    PathFragment lower = PathFragment.create("foo/header.h");
    PathFragment upper = PathFragment.create("foo/Header.h");
    Artifact.SourceArtifact lowerArtifact = artifactFactory.getSourceArtifact(lower, clientRoot);
    Artifact.SourceArtifact upperArtifact = artifactFactory.getSourceArtifact(upper, clientRoot);

    ImmutableList<Artifact.SourceArtifact> resultFromLower =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(lower, MAIN);
    ImmutableList<Artifact.SourceArtifact> resultFromUpper =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(upper, MAIN);

    assertThat(resultFromLower).containsExactly(lowerArtifact, upperArtifact);
    assertThat(resultFromUpper).containsExactly(lowerArtifact, upperArtifact);
  }

  @Test
  public void testCaseInsensitiveLookupWithThreeVariants() {
    artifactFactory.noteAnalysisStarting();
    PathFragment path1 = PathFragment.create("foo/File.h");
    PathFragment path2 = PathFragment.create("foo/file.h");
    PathFragment path3 = PathFragment.create("foo/FILE.h");
    Artifact.SourceArtifact a1 = artifactFactory.getSourceArtifact(path1, clientRoot);
    Artifact.SourceArtifact a2 = artifactFactory.getSourceArtifact(path2, clientRoot);
    Artifact.SourceArtifact a3 = artifactFactory.getSourceArtifact(path3, clientRoot);

    assertThat(artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(path1, MAIN))
        .containsExactly(a1, a2, a3);
    assertThat(artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(path2, MAIN))
        .containsExactly(a1, a2, a3);
    assertThat(
            artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(
                PathFragment.create("foo/fIlE.h"), MAIN))
        .containsExactly(a1, a2, a3);
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_derivedPathReturnsEmpty() {
    artifactFactory.noteAnalysisStarting();
    PathFragment derivedPath = PathFragment.create("bazel-out/x/bin/foo/header.h");

    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(derivedPath, MAIN);

    assertThat(result).isEmpty();
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_staleArtifactRevalidatedViaSourceRoot() {
    // First build: create an artifact.
    artifactFactory.noteAnalysisStarting();
    PathFragment path = PathFragment.create("foo/stale.h");
    var unused = artifactFactory.getSourceArtifact(path, clientRoot);

    // Second build: the artifact from the first build is invalid in the cache, but the method
    // falls back to source root resolution which re-validates it.
    artifactFactory.noteAnalysisStarting();
    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(path, MAIN);

    assertThat(result).hasSize(1);
    assertThat(result.get(0).getExecPath()).isEqualTo(path);
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_uplevelReturnsEmpty() {
    artifactFactory.noteAnalysisStarting();
    PathFragment uplevelPath = PathFragment.create("../outside/header.h");

    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(uplevelPath, MAIN);

    assertThat(result).isEmpty();
  }

  @Test
  public void testResolveSourceArtifactCaseInsensitively_fallbackToSourceRootResolution() {
    artifactFactory.noteAnalysisStarting();
    // Path not in cache but resolvable via source roots (foo package exists).
    PathFragment path = PathFragment.create("foo/brand_new.h");

    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(path, MAIN);

    assertThat(result).hasSize(1);
    assertThat(result.get(0).getExecPath()).isEqualTo(path);
  }

  @Test
  public void testExactLookupStillWorksWithCaseInsensitiveCache() {
    artifactFactory.noteAnalysisStarting();
    PathFragment lower = PathFragment.create("foo/header.h");
    Artifact.SourceArtifact lowerArtifact = artifactFactory.getSourceArtifact(lower, clientRoot);

    // Exact-case resolveSourceArtifact should return the correct artifact.
    assertThat(artifactFactory.resolveSourceArtifact(lower, MAIN)).isSameInstanceAs(lowerArtifact);
  }

  @Test
  public void
      testResolveSourceArtifactCaseInsensitively_staleArtifactWithDifferentCasingRevalidated() {
    // First build: create an artifact with specific casing.
    artifactFactory.noteAnalysisStarting();
    PathFragment originalPath = PathFragment.create("foo/Header.h");
    Artifact.SourceArtifact original = artifactFactory.getSourceArtifact(originalPath, clientRoot);

    // Second build: the artifact from the first build is stale. Resolve with different casing.
    artifactFactory.noteAnalysisStarting();
    PathFragment wrongCasePath = PathFragment.create("foo/header.h");
    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(wrongCasePath, MAIN);

    // Should return the original artifact with correct casing, not a new one.
    assertThat(result).hasSize(1);
    assertThat(result.get(0).getExecPath()).isEqualTo(originalPath);
    assertThat(result.get(0)).isSameInstanceAs(original);
  }

  @Test
  public void
      testResolveSourceArtifactCaseInsensitively_multipleStaleArtifactsWithDifferentCasingsRevalidated() {
    // First build: create artifacts with different casings.
    artifactFactory.noteAnalysisStarting();
    PathFragment path1 = PathFragment.create("foo/Header.h");
    PathFragment path2 = PathFragment.create("foo/HEADER.h");
    Artifact.SourceArtifact artifact1 = artifactFactory.getSourceArtifact(path1, clientRoot);
    Artifact.SourceArtifact artifact2 = artifactFactory.getSourceArtifact(path2, clientRoot);

    // Second build: both are stale. Resolve with yet another casing.
    artifactFactory.noteAnalysisStarting();
    PathFragment queryCasePath = PathFragment.create("foo/header.h");
    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(queryCasePath, MAIN);

    // Both original artifacts should be revalidated and returned.
    assertThat(result).containsExactly(artifact1, artifact2);
  }

  @Test
  public void testClearResetsCaseInsensitiveCache() {
    artifactFactory.noteAnalysisStarting();
    PathFragment path = PathFragment.create("foo/header.h");
    Artifact.SourceArtifact oldArtifact = artifactFactory.getSourceArtifact(path, clientRoot);

    artifactFactory.clear();
    setupRoots();
    artifactFactory.noteAnalysisStarting();

    Artifact.SourceArtifact newArtifact = artifactFactory.getSourceArtifact(path, clientRoot);
    assertThat(newArtifact).isNotSameInstanceAs(oldArtifact);
    ImmutableList<Artifact.SourceArtifact> result =
        artifactFactory.resolveSourceArtifactsAsciiCaseInsensitively(path, MAIN);
    assertThat(result).containsExactly(newArtifact);
  }

  private static class MockPackageRootResolver implements PackageRootResolver {
    private final Map<PathFragment, Root> packageRoots = Maps.newHashMap();

    public void setPackageRoots(Map<PackageIdentifier, Root> packageRoots) {
      for (Map.Entry<PackageIdentifier, Root> packageRoot : packageRoots.entrySet()) {
        this.packageRoots.put(packageRoot.getKey().getPackageFragment(), packageRoot.getValue());
      }
    }

    @Override
    public Map<PathFragment, Root> findPackageRootsForFiles(Iterable<PathFragment> execPaths) {
      Map<PathFragment, Root> result = new HashMap<>();
      for (PathFragment execPath : execPaths) {
        for (PathFragment dir = execPath.getParentDirectory(); dir != null;
            dir = dir.getParentDirectory()) {
          if (packageRoots.get(dir) != null) {
            result.put(execPath, packageRoots.get(dir));
          }
        }
        if (result.get(execPath) == null) {
          result.put(execPath, null);
        }
      }
      return result;
    }
  }
}
