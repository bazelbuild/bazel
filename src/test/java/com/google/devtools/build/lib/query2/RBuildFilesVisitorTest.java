// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.FileStateValue.DIRECTORY_FILE_STATE_NODE;
import static com.google.devtools.build.lib.skyframe.SkyFunctions.PACKAGE_LOOKUP;
import static com.google.devtools.build.lib.vfs.FileStateKey.FILE_STATE;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.skyframe.DirectoryListingStateValue;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RBuildFilesVisitor}. */
@RunWith(JUnit4.class)
public final class RBuildFilesVisitorTest {
  WalkableGraph graph;
  private static final FileSystem fs =
      new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
  private static final Root root = Root.fromPath(fs.getPath("/root/"));

  @Before
  public void setUp() {
    graph = mock(WalkableGraph.class);
  }

  @Test
  public void testPathFragmentToSkyKey_singleAncestor() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar"],
    // /*includeAncestorDirs=*/ false).

    // File "foo/bar" belongs in the same directory as "foo/BUILD"
    //
    // An empty existingDirs is passed to set includeAncestorDirs = false.

    Set<SkyKey> keys = getSkyKeysForFiles(existingPkgs("foo"), existingDirs(), diff("foo/bar"));
    assertThat(keys)
        .containsExactlyElementsIn(Iterables.concat(fileStates("foo/bar"), files("foo/bar")));
  }

  @Test
  public void testPathFragmentToSkyKey_singleAncestorTwoFiles() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar", "foo/baz"],
    // /*includeAncestorDirs=*/ false).
    //
    // File "foo/bar" and "foo/baz" both belong in the same directory as "foo/BUILD"
    //
    // An empty existingDirs is passed to set includeAncestorDirs = false.
    Set<SkyKey> keys =
        getSkyKeysForFiles(existingPkgs("foo"), existingDirs(), diff("foo/bar", "foo/baz"));

    assertThat(keys)
        .containsExactlyElementsIn(
            Iterables.concat(fileStates("foo/bar", "foo/baz"), files("foo/bar", "foo/baz")));
    // Because foo/bar and foo/baz belong in the same folder, we expect the package lookup to occur
    // at the same time and only once.
    verify(graph).getSuccessfulValues(any());
  }

  @Test
  public void testPathFragmentToSkyKey_packageNotFoundInDirectory() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar/baz", "foo/bar/bax"],
    // /*includeAncestorDirs=*/ false).
    //
    // File "foo/bar/baz" and "foo/bar/bax" both belong in a subdirectory of package foo.
    //
    // An empty existingDirs is passed to set includeAncestorDirs = false.
    Set<SkyKey> keys =
        getSkyKeysForFiles(existingPkgs("foo"), existingDirs(), diff("foo/bar/baz", "foo/bar/bax"));
    assertThat(keys)
        .containsExactlyElementsIn(
            Iterables.concat(
                fileStates("foo/bar/baz", "foo/bar/bax"), files("foo/bar/baz", "foo/bar/bax")));
    // We expect to take two steps of searching parent directories to find the package foo.
    verify(graph, times(2)).getSuccessfulValues(any());
  }

  @Test
  public void testPathFragmentToSkyKey_onlyAncestorPackageAndDirExists() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar/baz", "foo/bar/bax"],
    // /*includeAncestorDirs=*/ true).
    //
    // File "foo/bar/baz" and "foo/bar/bax" both belong in a subdirectory of package foo.
    //
    // existingDirs = ["foo"] means we are passing in true for 'includeAncestorDirs' and that
    // "foo/bar" is a newly created directory whereas "foo" already existed as a directory.
    Set<SkyKey> keys =
        getSkyKeysForFiles(
            existingPkgs("foo"), existingDirs("foo"), diff("foo/bar/baz", "foo/bar/bax"));

    // Because "foo/bar" is newly created, add a file and file state key for that directory as well
    // as adding the keys for the directory listing and directory listing state for "foo/bar" and
    // "foo".
    assertThat(keys)
        .containsExactlyElementsIn(
            Iterables.concat(
                fileStates("foo/bar/baz", "foo/bar/bax", "foo/bar"),
                files("foo/bar/baz", "foo/bar/bax", "foo/bar"),
                dirs("foo", "foo/bar"),
                dirStates("foo", "foo/bar")));
  }

  @Test
  public void testPathFragmentToSkyKey_onlyOneSubdirectoryExists() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar/f1", "foo/baz/f2"],
    // /*includeAncestorDirs=*/ true).
    //
    // File "foo/bar/f1" and "foo/baz/f2" both belong in a subdirectory of package foo.
    //
    // existingDirs = ["foo", "foo/bar"] means we are passing in true for 'includeAncestorDirs' and
    // that while "foo/bar" was an existing directory, "foo/baz" is newly created.
    Set<SkyKey> keys =
        getSkyKeysForFiles(
            existingPkgs("foo"), existingDirs("foo", "foo/bar"), diff("foo/bar/f1", "foo/baz/f2"));

    // We include a file and file state key for the newly added directory "foo/baz" but not for
    // "foo/bar". Because includeAncestorDirs was set to true, we also get directory listing and
    // directory listing state keys for all directories for which this could have changed.
    assertThat(keys)
        .containsExactlyElementsIn(
            Iterables.concat(
                fileStates("foo/bar/f1", "foo/baz/f2", "foo/baz"),
                files("foo/bar/f1", "foo/baz/f2", "foo/baz"),
                dirs("foo", "foo/bar", "foo/baz"),
                dirStates("foo", "foo/bar", "foo/baz")));
  }

  @Test
  public void testPathFragmentToSkyKey_bothSubdirectoryExists() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar/f1", "foo/baz/f2"],
    // /*includeAncestorDirs=*/ true).
    //
    // File "foo/bar/f1" and "foo/baz/f2" both belong in a subdirectory of package foo.
    //
    // existingDirs = ["foo", "foo/bar", "foo/baz"] means we are passing in true for
    // 'includeAncestorDirs' and that no new directories were created.
    Set<SkyKey> keys =
        getSkyKeysForFiles(
            existingPkgs("foo"),
            existingDirs("foo", "foo/bar", "foo/baz"),
            diff("foo/bar/f1", "foo/baz/f2"));

    // Since no new directories were created, we expect no file or file state keys for them. Because
    // includeAncestorKeys was set to true, include the directory listing and directory listing
    // state keys of the two existing directories that had files in the diff.
    assertThat(keys)
        .containsExactlyElementsIn(
            Iterables.concat(
                fileStates("foo/bar/f1", "foo/baz/f2"),
                files("foo/bar/f1", "foo/baz/f2"),
                dirs("foo/bar", "foo/baz"),
                dirStates("foo/bar", "foo/baz")));
  }

  @Test
  public void testPathFragmentToSkyKey_packageInDifferentAncestor() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar", "foo/bar/bax"],
    // /*includeAncestorDirs=*/ false).
    //
    // File "foo/bar" and "foo/bar/bax" belong in subdirectories with a differing amount of nesting
    // under package foo.
    //
    // An empty existingDirs is passed to set includeAncestorDirs = false.
    Set<SkyKey> keys =
        getSkyKeysForFiles(existingPkgs("foo"), existingDirs(), diff("foo/bar", "foo/bar/bax"));
    assertThat(keys)
        .containsExactlyElementsIn(
            Iterables.concat(
                fileStates("foo/bar", "foo/bar/bax"), files("foo/bar", "foo/bar/bax")));
    // Because we expect the search for "foo/bar/bax"'s package to take two hops, we expect two
    // calls to the graph in the package search.
    verify(graph, times(2)).getSuccessfulValues(any());
  }

  @Test
  public void testPathFragmentToSkyKey_noAncestorKeys() throws Exception {
    // Tests RBuildFilesVisitor#getSkyKeysForFileFragments(
    // graph,
    // /*files=*/ ["foo/bar"],
    // /*includeAncestorDirs=*/ false).
    //
    // File "foo/bar" has no parent package
    //
    // An empty existingDirs is passed to set includeAncestorDirs = false.
    Set<SkyKey> keys = getSkyKeysForFiles(existingPkgs(), existingDirs(), diff("foo/bar"));

    // Because "foo/bar" has no parent package, we are not able to return any keys.
    assertThat(keys).isEmpty();
  }

  /**
   * Calls RBuildFilesVisitor#getSkyKeysForFileFragments where the files passed in are specified by
   * the 'diff' variable.
   *
   * <p>The aforementioned function makes a skyframe call to retrieve PackageLookupValues and
   * FileStateValues and so the parameters 'existingPackages' and 'existingDirectories' allows us to
   * seed our mock graph with successful package lookups and existent directory file states for the
   * specified paths.
   *
   * <p>Note: The skyframe lookups for FileStateValues occurs only if the parameter
   * 'includeAncestorKeys' in RBuildFilesVisitor#getSkyKeysForFileFragments is true and so the paths
   * inside 'existingDirectories' are relevant if and only if 'includeAncestorKeys' is true. Because
   * of this, we pass in 'includeAncestorKeys' as true if and only if 'existingDirectories' is
   * non-empty.
   *
   * <p>Note: The skyquery function 'rbuildfiles' uses RBuildFilesVisitor#getSkyKeysForFileFragments
   * with 'includeAncestorKeys' as being false. Passing in an empty set for 'existingDirs' allows
   * this mode of operation to be tested.
   */
  private Set<SkyKey> getSkyKeysForFiles(
      Set<PathFragment> existingPackages,
      Set<PathFragment> existingDirectories,
      Set<PathFragment> pathFragments)
      throws Exception {
    when(graph.getSuccessfulValues(any()))
        .thenAnswer(
            invocationOnMock -> {
              Map<SkyKey, SkyValue> result = new HashMap<>();
              Iterable<?> paths = (Iterable<?>) invocationOnMock.getArgument(0);
              for (Object object : paths) {
                assertThat(object).isInstanceOf(SkyKey.class);
                SkyKey key = (SkyKey) object;
                if (key.functionName().equals(PACKAGE_LOOKUP)) {
                  PathFragment fragment = ((PackageIdentifier) key.argument()).getPackageFragment();
                  if (existingPackages.contains(fragment)) {
                    result.put(key, PackageLookupValue.success(root, BuildFileName.BUILD));
                  }
                } else if (key.functionName().equals(FILE_STATE)) {
                  PathFragment fragment = ((RootedPath) key.argument()).getRootRelativePath();
                  if (existingDirectories.contains(fragment)) {
                    result.put(key, DIRECTORY_FILE_STATE_NODE);
                  }
                } else {
                  throw new IllegalStateException("Unexpected skyframe lookup: " + key);
                }
              }
              return result;
            });

    return RBuildFilesVisitor.getSkyKeysForFileFragments(
        graph, pathFragments, !existingDirectories.isEmpty());
  }

  private static Set<SkyKey> fileStates(String... paths) {
    return makeKeys(FileStateValue::key, paths);
  }

  private static Set<SkyKey> files(String... paths) {
    return makeKeys(FileValue::key, paths);
  }

  private static Set<SkyKey> dirStates(String... paths) {
    return makeKeys(DirectoryListingStateValue::key, paths);
  }

  private static Set<SkyKey> dirs(String... paths) {
    return makeKeys(DirectoryListingValue::key, paths);
  }

  private static Set<SkyKey> makeKeys(
      Function<RootedPath, SkyKey> rootedPathToKey, String... paths) {
    return toPaths(paths).stream()
        .map(path -> rootedPathToKey.apply(RootedPath.toRootedPath(root, path)))
        .collect(Collectors.toSet());
  }

  private static Set<PathFragment> diff(String... files) {
    return toPaths(files);
  }

  private static Set<PathFragment> existingDirs(String... files) {
    return toPaths(files);
  }

  private static Set<PathFragment> existingPkgs(String... files) {
    return toPaths(files);
  }

  private static Set<PathFragment> toPaths(String... files) {
    Set<PathFragment> result = new HashSet<>();
    for (String file : files) {
      result.add(PathFragment.create(file));
    }
    return result;
  }
}
