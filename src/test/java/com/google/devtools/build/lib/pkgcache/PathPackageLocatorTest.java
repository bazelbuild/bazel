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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test package-path logic.
 */
@RunWith(JUnit4.class)
public class PathPackageLocatorTest extends FoundationTestCase {
  private Path buildBazelFile1A;
  private Path buildFile1B;
  private Path buildFile2C;
  private Path buildFile2CD;
  private Path buildFile2F;
  private Path buildFile2FGH;
  private Path buildBazelFile3A;
  private Path buildFile3B;
  private Path buildFile3CI;
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

  protected PathPackageLocator getLocator() {
    return locator;
  }

  @Before
  public final void createFiles() throws Exception {
    // Root 1:
    //   WORKSPACE
    //   /A/BUILD.bazel // This is the actual buildfile for this package.
    //   /A/BUILD       // This is a dummy buildfile and isn't used.
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
    buildBazelFile1A = createBuildFile(rootDir1, "A", true);
    buildFile1B = createBuildFile(rootDir1, "B");
    createBuildFile(rootDir1, "C/I");
    scratch.file(rootDir1.getPathString() + "/F/G");

    rootDir1.getRelative("C").createDirectory();
    rootDir1.getRelative("C/D").createDirectory();
    rootDir1.getRelative("C/E").createDirectory();

    // Workspace file in rootDir2.
    scratch.file(rootDir2 + "/WORKSPACE");
    createBuildFile(rootDir2, "B");
    buildFile2C = createBuildFile(rootDir2, "C");
    buildFile2CD = createBuildFile(rootDir2, "C/D");
    buildFile2F = createBuildFile(rootDir2, "F");
    buildFile2FGH = createBuildFile(rootDir2, "F/G/H");
    scratch.file(rootDir2.getPathString() + "/C/I");

    // Root3 just needs a symlink to 4
    FileSystemUtils.ensureSymbolicLink(
        rootDir3ParentParent.getRelative("READONLY"), rootDir4Parent);
    buildBazelFile3A = rootDir3.getRelative("A/BUILD.bazel");
    buildFile3B = rootDir3.getRelative("B/BUILD");
    buildFile3CI = rootDir3.getRelative("C/I/BUILD");

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

    locator =
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(Root.fromPath(rootDir1), Root.fromPath(rootDir2)),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    locatorWithSymlinks =
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(Root.fromPath(rootDir3)),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
  }

  private Path createBuildFile(Path workspace, String packageName) throws IOException {
    return createBuildFile(workspace, packageName, false);
  }

  private Path createBuildFile(Path workspace, String packageName, boolean dotBazel)
      throws IOException {
    String buildFileName = dotBazel ? "BUILD.bazel" : "BUILD";
    return scratch.file(workspace + "/" + packageName + "/" + buildFileName);
  }

  private void checkFails(String packageName, String expectedError) {
    checkFails(getLocator(), packageName, expectedError);
  }

  private static void checkFails(
      PathPackageLocator locator, String packageName, String expectedError) {
    NoSuchPackageException e =
        assertThrows(
            NoSuchPackageException.class,
            () -> locator.getPackageBuildFile(PackageIdentifier.createInMainRepo(packageName)));
    assertThat(e).hasMessageThat().ignoringCase().contains(expectedError);
    assertThat(e.getDetailedExitCode().getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(PackageLoading.Code.BUILD_FILE_MISSING);
  }

  @Test
  public void testGetPackageBuildFile() throws Exception {
    AtomicReference<? extends UnixGlob.FilesystemCalls> cache = UnixGlob.DEFAULT_SYSCALLS_REF;
    assertThat(locator.getPackageBuildFile(PackageIdentifier.createInMainRepo("A")))
        .isEqualTo(buildBazelFile1A);
    assertThat(locator.getPackageBuildFileNullable(PackageIdentifier.createInMainRepo("A"), cache))
        .isEqualTo(buildBazelFile1A);
    assertThat(locator.getPackageBuildFile(PackageIdentifier.createInMainRepo("B")))
        .isEqualTo(buildFile1B);
    assertThat(locator.getPackageBuildFileNullable(PackageIdentifier.createInMainRepo("B"), cache))
        .isEqualTo(buildFile1B);
    assertThat(locator.getPackageBuildFile(PackageIdentifier.createInMainRepo("C")))
        .isEqualTo(buildFile2C);
    assertThat(locator.getPackageBuildFileNullable(PackageIdentifier.createInMainRepo("C"), cache))
        .isEqualTo(buildFile2C);
    assertThat(locator.getPackageBuildFile(PackageIdentifier.createInMainRepo("C/D")))
        .isEqualTo(buildFile2CD);
    assertThat(
            locator.getPackageBuildFileNullable(PackageIdentifier.createInMainRepo("C/D"), cache))
        .isEqualTo(buildFile2CD);
    checkFails("C/E",
               "no such package 'C/E': BUILD file not found on package path");
    assertThat(
            locator.getPackageBuildFileNullable(PackageIdentifier.createInMainRepo("C/E"), cache))
        .isNull();
    assertThat(locator.getPackageBuildFile(PackageIdentifier.createInMainRepo("F")))
        .isEqualTo(buildFile2F);
    checkFails("F/G",
               "no such package 'F/G': BUILD file not found on package path");
    assertThat(
            locator.getPackageBuildFileNullable(PackageIdentifier.createInMainRepo("F/G"), cache))
        .isNull();
    assertThat(locator.getPackageBuildFile(PackageIdentifier.createInMainRepo("F/G/H")))
        .isEqualTo(buildFile2FGH);
    assertThat(
            locator.getPackageBuildFileNullable(PackageIdentifier.createInMainRepo("F/G/H"), cache))
        .isEqualTo(buildFile2FGH);
    checkFails("I", "no such package 'I': BUILD file not found on package path");
  }

  @Test
  public void testGetPackageBuildFileWithSymlinks() throws Exception {
    assertThat(locatorWithSymlinks.getPackageBuildFile(PackageIdentifier.createInMainRepo("A")))
        .isEqualTo(buildBazelFile3A);
    assertThat(locatorWithSymlinks.getPackageBuildFile(PackageIdentifier.createInMainRepo("B")))
        .isEqualTo(buildFile3B);
    assertThat(locatorWithSymlinks.getPackageBuildFile(PackageIdentifier.createInMainRepo("C/I")))
        .isEqualTo(buildFile3CI);
    checkFails(
        locatorWithSymlinks, "C/D", "no such package 'C/D': BUILD file not found on package path");
  }

  @Test
  public void testGetWorkspaceFile() throws Exception {
    assertThat(locator.getWorkspaceFile()).isEqualTo(rootDir1WorkspaceFile);
  }

  private Path setLocator(String root) {
    Path nonExistentRoot = scratch.resolve(root);
    this.locator =
        PathPackageLocator.create(
            /*outputBase=*/ null,
            Arrays.asList(root),
            reporter,
            /*workspace=*/ FileSystemUtils.getWorkingDirectory(),
            /* clientWorkingDirectory= */ FileSystemUtils.getWorkingDirectory(
                scratch.getFileSystem()),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    return nonExistentRoot;
  }

  @Test
  public void testExists() throws Exception {
    Path nonExistentRoot1 = setLocator("/non/existent/1/workspace");
    // Now let's create the root:
    createBuildFile(nonExistentRoot1, "X");
    // The package isn't found
    // The package is found, because we didn't drop the root:
    assertThrows(
        NoSuchPackageException.class,
        () -> locator.getPackageBuildFile(PackageIdentifier.createInMainRepo("X")));

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
    assertThat(
            PathPackageLocator.create(
                    /*outputBase=*/ null,
                    pathElements,
                    reporter,
                    workspace.asFragment(),
                    clientPath,
                    BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY)
                .getPathEntries())
        .containsExactly(
            Root.fromPath(belowClient),
            Root.fromPath(clientPath),
            Root.fromPath(workspace.getRelative("somewhere")),
            Root.fromPath(clientPath.getRelative("below")))
        .inOrder();
  }

  @Test
  public void testRelativePathWarning() throws Exception {
    Path workspace = scratch.dir("/some/path/to/workspace");

    // No warning if workspace == cwd.
    PathPackageLocator.create(
        /*outputBase=*/ null,
        ImmutableList.of("./foo"),
        reporter,
        workspace.asFragment(),
        workspace,
        BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    assertThat(eventCollector.count()).isSameInstanceAs(0);

    PathPackageLocator.create(
        /*outputBase=*/ null,
        ImmutableList.of("./foo"),
        reporter,
        workspace.asFragment(),
        workspace.getRelative("foo"),
        BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    assertThat(eventCollector.count()).isSameInstanceAs(1);
    assertContainsEvent("The package path element 'foo' will be taken relative");
  }

  /** Regression test for bug: "IllegalArgumentException in PathPackageLocator.create()" */
  @Test
  public void testDollarSigns() throws Exception {
    Path workspace = scratch.dir("/some/path/to/workspace$1");

    PathPackageLocator.create(
        /*outputBase=*/ null,
        ImmutableList.of("%workspace%/blabla"),
        reporter,
        workspace.asFragment(),
        workspace.getRelative("foo"),
        BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
  }
}
