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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ARTIFACT_OWNER;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Tests {@link ArtifactFactory}. Also see {@link ArtifactTest} for a test
 * of individual artifacts.
 */
@RunWith(JUnit4.class)
public class ArtifactFactoryTest {

  private Scratch scratch = new Scratch();

  private Path execRoot;
  private Root clientRoot;
  private Root clientRoRoot;
  private Root alienRoot;
  private Root outRoot;

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

  @Before
  public final void createFiles() throws Exception  {
    execRoot = scratch.dir("/output/workspace");
    clientRoot = Root.asSourceRoot(scratch.dir("/client/workspace"));
    clientRoRoot = Root.asSourceRoot(scratch.dir("/client/RO/workspace"));
    alienRoot = Root.asSourceRoot(scratch.dir("/client/workspace"));
    outRoot = Root.asDerivedRoot(execRoot, execRoot.getRelative("out-root/x/bin"));

    fooPath = new PathFragment("foo");
    fooPackage = PackageIdentifier.createInDefaultRepo(fooPath);
    fooRelative = fooPath.getRelative("foosource.txt");

    barPath = new PathFragment("foo/bar");
    barPackage = PackageIdentifier.createInDefaultRepo(barPath);
    barRelative = barPath.getRelative("barsource.txt");

    alienPath = new PathFragment("external/alien");
    alienPackage = PackageIdentifier.create("@alien", alienPath);
    alienRelative = alienPath.getRelative("alien.txt");

    artifactFactory = new ArtifactFactory(execRoot);
    setupRoots();
  }

  private void setupRoots() {
    Map<PackageIdentifier, Root> packageRootMap = new HashMap<>();
    packageRootMap.put(fooPackage, clientRoot);
    packageRootMap.put(barPackage, clientRoRoot);
    packageRootMap.put(alienPackage, alienRoot);
    artifactFactory.setPackageRoots(packageRootMap);
    artifactFactory.setDerivedArtifactRoots(ImmutableList.of(outRoot));
  }

  @Test
  public void testGetSourceArtifactYieldsSameArtifact() throws Exception {
    assertSame(artifactFactory.getSourceArtifact(fooRelative, clientRoot),
               artifactFactory.getSourceArtifact(fooRelative, clientRoot));
  }

  @Test
  public void testGetSourceArtifactUnnormalized() throws Exception {
    assertSame(artifactFactory.getSourceArtifact(fooRelative, clientRoot),
               artifactFactory.getSourceArtifact(new PathFragment("foo/./foosource.txt"),
                   clientRoot));
  }

  @Test
  public void testResolveArtifact_noDerived_simpleSource() throws Exception {
    assertSame(artifactFactory.getSourceArtifact(fooRelative, clientRoot),
        artifactFactory.resolveSourceArtifact(fooRelative));
    assertSame(artifactFactory.getSourceArtifact(barRelative, clientRoRoot),
        artifactFactory.resolveSourceArtifact(barRelative));
  }

  @Test
  public void testResolveArtifact_inExternalRepo() throws Exception {
    assertSame(
        artifactFactory.getSourceArtifact(alienRelative, alienRoot),
        artifactFactory.resolveSourceArtifact(alienRelative));
  }

  @Test
  public void testResolveArtifact_noDerived_derivedRoot() throws Exception {
    assertNull(artifactFactory.resolveSourceArtifact(
            outRoot.getPath().getRelative(fooRelative).relativeTo(execRoot)));
    assertNull(artifactFactory.resolveSourceArtifact(
            outRoot.getPath().getRelative(barRelative).relativeTo(execRoot)));
  }

  @Test
  public void testResolveArtifact_noDerived_simpleSource_other() throws Exception {
    Artifact actual = artifactFactory.resolveSourceArtifact(fooRelative);
    assertSame(artifactFactory.getSourceArtifact(fooRelative, clientRoot), actual);
    actual = artifactFactory.resolveSourceArtifact(barRelative);
    assertSame(artifactFactory.getSourceArtifact(barRelative, clientRoRoot), actual);
  }

  @Test
  public void testResolveArtifactWithUpLevelFailsCleanly() throws Exception {
    // We need a package in the root directory to make every exec path (even one with up-level
    // references) be in a package.
    Map<PackageIdentifier, Root> packageRoots = ImmutableMap.of(
        PackageIdentifier.createInDefaultRepo(new PathFragment("")), clientRoot);
    artifactFactory.setPackageRoots(packageRoots);
    PathFragment outsideWorkspace = new PathFragment("../foo");
    PathFragment insideWorkspace =
        new PathFragment("../" + clientRoot.getPath().getBaseName() + "/foo");
    assertNull(artifactFactory.resolveSourceArtifact(outsideWorkspace));
    assertNull("Up-level-containing paths that descend into the right workspace aren't allowed",
            artifactFactory.resolveSourceArtifact(insideWorkspace));
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
    assertNotSame(fooArtifact, artifactFactory.getSourceArtifact(fooRelative, clientRoot));
  }

  @Test
  public void testFindDerivedRoot() throws Exception {
    assertSame(outRoot,
        artifactFactory.findDerivedRoot(outRoot.getPath().getRelative(fooRelative)));
    assertSame(outRoot,
        artifactFactory.findDerivedRoot(outRoot.getPath().getRelative(barRelative)));
  }

  @Test
  public void testSetGeneratingActionIdempotenceNewActionGraph() throws Exception {
    Artifact a = artifactFactory.getDerivedArtifact(fooRelative, outRoot, NULL_ARTIFACT_OWNER);
    Artifact b = artifactFactory.getDerivedArtifact(barRelative, outRoot, NULL_ARTIFACT_OWNER);
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    Action originalAction = new ActionsTestUtil.NullAction(NULL_ACTION_OWNER, a);
    actionGraph.registerAction(originalAction);

    // Creating a second Action referring to the Artifact should create a conflict.
    try {
      Action action = new ActionsTestUtil.NullAction(NULL_ACTION_OWNER, a, b);
      actionGraph.registerAction(action);
      fail();
    } catch (ActionConflictException e) {
      assertSame(a, e.getArtifact());
      assertSame(originalAction, actionGraph.getGeneratingAction(a));
    }
  }

  @Test
  public void testGetDerivedArtifact() throws Exception {
    PathFragment toolPath = new PathFragment("_bin/tool");
    Artifact artifact = artifactFactory.getDerivedArtifact(toolPath);
    assertEquals(toolPath, artifact.getExecPath());
    assertEquals(Root.asDerivedRoot(execRoot), artifact.getRoot());
    assertEquals(execRoot.getRelative(toolPath), artifact.getPath());
    assertNull(artifact.getOwner());
  }

  @Test
  public void testGetDerivedArtifactFailsForAbsolutePath() throws Exception {
    try {
      artifactFactory.getDerivedArtifact(new PathFragment("/_bin/b"));
      fail();
    } catch (IllegalArgumentException e) {
      // Expected exception
    }
  }

  private static class MockPackageRootResolver implements PackageRootResolver {
    private Map<PathFragment, Root> packageRoots = Maps.newHashMap();

    public void setPackageRoots(Map<PackageIdentifier, Root> packageRoots) {
      for (Entry<PackageIdentifier, Root> packageRoot : packageRoots.entrySet()) {
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
