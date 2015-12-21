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
package com.google.devtools.build.lib.pkgcache;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixGlob;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Test package-path logic.
 */
@RunWith(JUnit4.class)
public class PathPackageLocatorTest extends FoundationTestCase {
  private Path buildFile_1A;
  private Path buildFile_1B;
  private Path buildFile_2C;
  private Path buildFile_2CD;
  private Path buildFile_2F;
  private Path buildFile_2FGH;
  private Path buildFile_3A;
  private Path buildFile_3B;
  private Path buildFile_3CI;
  private Path rootDir1;
  private Path rootDir1WorkspaceFile;
  private Path rootDir2;
  private Path rootDir3ParentParent;
  private Path rootDir3;
  private Path rootDir4Parent;
  private Path rootDir4;
  private Path rootDir5;
  private PathPackageLocator locator;
  private PathPackageLocator locatorWithSymlinks;

  protected PathPackageLocator getLocator() { return locator; }


  @Before
  public final void createFiles() throws Exception {
    // Root 1:
    //   WORKSPACE
    //   /A/BUILD
    //   /B/BUILD
    //   /C/I/BUILD
    //   /C/D
    //   /C/E
    //   /F/G         // This is a file, not a directory.
    //
    // Root 2:
    //   WORKSPACE
    //   /B/BUILD
    //   /C/BUILD
    //   /C/D/BUILD
    //   /F/BUILD
    //   /F/G
    //   /F/G/H/BUILD
    //   /I/BUILD         // This is a directory, not a file.
    //
    // Root 3:
    //   /usr/local/google/jrluser-foo/READONLY -> root4
    //
    // Root 4 (not used as a package root, but root 3 points to this)
    //   /A -> root1/A
    //   /B/BUILD -> root1/B/BUILD
    //   /C/I/BUILD -> root1/C/I/BUILD
    //   /C/D -> root1/C/D
    //   /C/E -> root1/C/E
    //   /F/G -> root1/F/G
    //   /H/I -> root5/H/I
    //
    // Root 5 (pointed to by Root 4)
    //   Note: the following BUILD file will be found if explicitly specified, but it
    //     would not be found if using wildcards.  That is because isDirectory
    //     will return false since the symlink target is not in the workspace.
    //   /H/I/BUILD
    rootDir1 = scratch.resolve("/home/user/src-foo/workspace");
    rootDir2 = scratch.resolve("/somewhere/1234567/build/workspace");
    rootDir3ParentParent = scratch.resolve("/usr/local/google/jrluser-foo");
    rootDir3 = rootDir3ParentParent.getRelative("READONLY/workspace");
    rootDir4Parent =  scratch.resolve("/usr/local/symlinks/client_symlink_jrluser-foo");
    rootDir4 = rootDir4Parent.getRelative("workspace");
    rootDir5 = scratch.resolve("/foo/bar");

    rootDir1WorkspaceFile = scratch.file(rootDir1 + "/WORKSPACE");
    buildFile_1A   = createBuildFile(rootDir1, "A");
    buildFile_1B   = createBuildFile(rootDir1, "B");
    createBuildFile(rootDir1, "C/I");
    scratch.file(rootDir1.getPathString() + "/F/G");

    rootDir1.getRelative("C").createDirectory();
    rootDir1.getRelative("C/D").createDirectory();
    rootDir1.getRelative("C/E").createDirectory();

    // Workspace file in rootDir2.
    scratch.file(rootDir2 + "/WORKSPACE");
    createBuildFile(rootDir2, "B");
    buildFile_2C   = createBuildFile(rootDir2, "C");
    buildFile_2CD  = createBuildFile(rootDir2, "C/D");
    buildFile_2F   = createBuildFile(rootDir2, "F");
    buildFile_2FGH = createBuildFile(rootDir2, "F/G/H");
    scratch.file(rootDir2.getPathString() + "/C/I");

    // Root3 just needs a symlink to 4
    FileSystemUtils.ensureSymbolicLink(
        rootDir3ParentParent.getRelative("READONLY"), rootDir4Parent);
    buildFile_3A = rootDir3.getRelative("A/BUILD");
    buildFile_3B = rootDir3.getRelative("B/BUILD");
    buildFile_3CI = rootDir3.getRelative("C/I/BUILD");

    // Root4
    FileSystemUtils.ensureSymbolicLink(
        rootDir4.getRelative("A"), rootDir1.getRelative("A"));
    FileSystemUtils.ensureSymbolicLink(
        rootDir4.getRelative("B/BUILD"), rootDir1.getRelative("B/BUILD"));
    FileSystemUtils.ensureSymbolicLink(
        rootDir4.getRelative("C/I/BUILD"), rootDir1.getRelative("C/I/BUILD"));
    FileSystemUtils.ensureSymbolicLink(
        rootDir4.getRelative("C/D/BUILD"), rootDir1.getRelative("C/D/BUILD"));
    FileSystemUtils.ensureSymbolicLink(
        rootDir4.getRelative("C/E/BUILD"), rootDir1.getRelative("C/E/BUILD"));
    FileSystemUtils.ensureSymbolicLink(
        rootDir4.getRelative("F/G/BUILD"), rootDir1.getRelative("F/G/BUILD"));
    FileSystemUtils.ensureSymbolicLink(
        rootDir4.getRelative("H/I"), rootDir5.getRelative("H/I"));

    // Root5
    createBuildFile(rootDir5, "H/I");

    locator = new PathPackageLocator(outputBase, ImmutableList.of(rootDir1, rootDir2));
    locatorWithSymlinks = new PathPackageLocator(outputBase, ImmutableList.of(rootDir3));
  }

  private Path createBuildFile(Path workspace, String packageName) throws IOException {
    return scratch.file(workspace + "/" + packageName + "/BUILD");
  }

  private void checkFails(String packageName, String expectorError) {
    try {
      getLocator().getPackageBuildFile(PackageIdentifier.createInDefaultRepo(packageName));
      fail();
    } catch (NoSuchPackageException e) {
      assertThat(e).hasMessage(expectorError);
    }
  }

  @Test
  public void testGetPackageBuildFile() throws Exception {
    AtomicReference<? extends UnixGlob.FilesystemCalls> cache = UnixGlob.DEFAULT_SYSCALLS_REF;
    assertEquals(buildFile_1A, locator.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("A")));
    assertEquals(buildFile_1A, locator.getPackageBuildFileNullable(
        PackageIdentifier.createInDefaultRepo("A"), cache));
    assertEquals(buildFile_1B, locator.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("B")));
    assertEquals(buildFile_1B, locator.getPackageBuildFileNullable(
        PackageIdentifier.createInDefaultRepo("B"), cache));
    assertEquals(buildFile_2C, locator.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("C")));
    assertEquals(buildFile_2C, locator.getPackageBuildFileNullable(
        PackageIdentifier.createInDefaultRepo("C"), cache));
    assertEquals(buildFile_2CD, locator.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("C/D")));
    assertEquals(buildFile_2CD, locator.getPackageBuildFileNullable(
        PackageIdentifier.createInDefaultRepo("C/D"), cache));
    checkFails("C/E",
               "no such package 'C/E': BUILD file not found on package path");
    assertNull(locator.getPackageBuildFileNullable(
        PackageIdentifier.createInDefaultRepo("C/E"), cache));
    assertEquals(buildFile_2F,
                 locator.getPackageBuildFile(PackageIdentifier.createInDefaultRepo("F")));
    checkFails("F/G",
               "no such package 'F/G': BUILD file not found on package path");
    assertNull(locator.getPackageBuildFileNullable(
        PackageIdentifier.createInDefaultRepo("F/G"), cache));
    assertEquals(buildFile_2FGH, locator.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("F/G/H")));
    assertEquals(buildFile_2FGH, locator.getPackageBuildFileNullable(
        PackageIdentifier.createInDefaultRepo("F/G/H"), cache));
    checkFails("I", "no such package 'I': BUILD file not found on package path");
  }

  @Test
  public void testGetPackageBuildFileWithSymlinks() throws Exception {
    assertEquals(buildFile_3A, locatorWithSymlinks.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("A")));
    assertEquals(buildFile_3B, locatorWithSymlinks.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("B")));
    assertEquals(buildFile_3CI, locatorWithSymlinks.getPackageBuildFile(
        PackageIdentifier.createInDefaultRepo("C/I")));
    try {
      locatorWithSymlinks.getPackageBuildFile(PackageIdentifier.createInDefaultRepo("C/D"));
      fail();
    } catch (BuildFileNotFoundException e) {
      assertThat(e).hasMessage("no such package 'C/D': BUILD file not found on package path");
    }
  }

  @Test
  public void testGetWorkspaceFile() throws Exception {
    assertEquals(rootDir1WorkspaceFile, locator.getWorkspaceFile());
  }

  private Path setLocator(String root) {
    Path nonExistentRoot = scratch.resolve(root);
    this.locator = PathPackageLocator.create(null, Arrays.asList(root), reporter,
            /*workspace=*/ FileSystemUtils.getWorkingDirectory(scratch.getFileSystem()),
            /*workingDir=*/ FileSystemUtils.getWorkingDirectory(scratch.getFileSystem()));
    return nonExistentRoot;
  }

  @Test
  public void testExists() throws Exception {
    Path nonExistentRoot1 = setLocator("/non/existent/1/workspace");
    // Now let's create the root:
    createBuildFile(nonExistentRoot1, "X");
    // The package isn't found
    // The package is found, because we didn't drop the root:
    try {
      locator.getPackageBuildFile(PackageIdentifier.createInDefaultRepo("X"));
      fail("Exception expected");
    } catch (NoSuchPackageException e) {
    }
    Path nonExistentRoot2 = setLocator("/non/existent/2/workspace");
    // Now let's create the root:
    createBuildFile(nonExistentRoot2, "X");
    // ...but the package is still not found, because we dropped the root:
    checkFails("X",
               "no such package 'X': BUILD file not found on package path");
  }

  @Test
  public void testPathResolution() throws Exception {
    Path workspace = scratch.dir("/some/path/to/workspace");
    Path clientPath = workspace.getRelative("somewhere/below/workspace");
    scratch.dir(clientPath.getPathString());
    Path belowClient = clientPath.getRelative("below/client");
    scratch.dir(belowClient.getPathString());

    List<String> pathElements = ImmutableList.of(
        "./below/client",        // Client-relative
        ".",                     // Client-relative
        "%workspace%/somewhere", // Workspace-relative
        // Absolute
        clientPath.getRelative("below").getPathString());
    assertThat(PathPackageLocator
        .create(null, pathElements, reporter, workspace, clientPath).getPathEntries())
        .containsExactly(belowClient, clientPath, workspace.getRelative("somewhere"),
            clientPath.getRelative("below")).inOrder();
  }

  @Test
  public void testRelativePathWarning() throws Exception {
    Path workspace = scratch.dir("/some/path/to/workspace");

    // No warning if workspace == cwd.
    PathPackageLocator.create(null, ImmutableList.of("./foo"), reporter, workspace, workspace);
    assertSame(0, eventCollector.count());

    PathPackageLocator.create(
        null, ImmutableList.of("./foo"), reporter, workspace, workspace.getRelative("foo"));
    assertSame(1, eventCollector.count());
    assertContainsEvent("The package path element './foo' will be taken relative");
  }

  /** Regression test for bug: "IllegalArgumentException in PathPackageLocator.create()" */
  @Test
  public void testDollarSigns() throws Exception {
    Path workspace = scratch.dir("/some/path/to/workspace$1");

    PathPackageLocator.create(null, ImmutableList.of("%workspace%/blabla"), reporter, workspace,
        workspace.getRelative("foo"));
  }
}
