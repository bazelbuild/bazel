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
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
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
    outRoot = ArtifactRoot.asDerivedRoot(execRoot, false, "out-root", "x", "bin");

    fooPath = PathFragment.create("foo");
    fooPackage = PackageIdentifier.createInMainRepo(fooPath);
    fooRelative = fooPath.getRelative("foosource.txt");

    barPath = PathFragment.create("foo/bar");
    barPackage = PackageIdentifier.createInMainRepo(barPath);
    barRelative = barPath.getRelative("barsource.txt");

    alienPath = PathFragment.create("external/alien");
    alienPackage = PackageIdentifier.create("@alien", alienPath);
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
