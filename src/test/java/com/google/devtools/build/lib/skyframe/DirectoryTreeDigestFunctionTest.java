// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionException;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionUniquenessFunction;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DirectoryTreeDigestFunctionTest extends FoundationTestCase {

  private RecordingDifferencer differencer;
  private ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  private EvaluationContext evaluationContext;

  @Before
  public void setup() {
    differencer = new SequencedRecordingDifferencer();
    evaluationContext =
        EvaluationContext.newBuilder().setParallelism(8).setEventHandler(reporter).build();
    AtomicReference<PathPackageLocator> packageLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, rootDirectory),
            rootDirectory,
            AnalysisMock.get().getProductName());
    ExternalFilesHelper externalFilesHelper =
        ExternalFilesHelper.createForTesting(
            packageLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);

    skyFunctions =
        ImmutableMap.<SkyFunctionName, SkyFunction>builder()
            .put(SkyFunctions.FILE, new FileFunction(packageLocator, directories))
            .put(
                FileStateKey.FILE_STATE,
                new FileStateFunction(
                    Suppliers.ofInstance(new TimestampGranularityMonitor(BlazeClock.instance())),
                    SyscallCache.NO_CACHE,
                    externalFilesHelper))
            .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
            .put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction())
            .put(
                SkyFunctions.DIRECTORY_LISTING_STATE,
                new DirectoryListingStateFunction(externalFilesHelper, SyscallCache.NO_CACHE))
            .put(SkyFunctions.DIRECTORY_TREE_DIGEST, new DirectoryTreeDigestFunction())
            .put(
                FileSymlinkInfiniteExpansionUniquenessFunction.NAME,
                new FileSymlinkInfiniteExpansionUniquenessFunction())
            .buildOrThrow();

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, packageLocator.get());
  }

  private String getTreeDigest(String path) throws Exception {
    return getTreeDigest(path, ImmutableList.of());
  }

  private String getTreeDigest(String path, ImmutableList<String> excludes) throws Exception {
    return getTreeDigest(
        RootedPath.toRootedPath(Root.absoluteRoot(fileSystem), scratch.resolve(path)), excludes);
  }

  private String getTreeDigest(RootedPath rootedPath, ImmutableList<String> excludes)
      throws Exception {
    return getTreeDigest(
        new InMemoryMemoizingEvaluator(skyFunctions, differencer), rootedPath, excludes);
  }

  private String getTreeDigest(
      MemoizingEvaluator evaluator, RootedPath rootedPath, ImmutableList<String> excludes)
      throws Exception {
    SkyKey key = DirectoryTreeDigestValue.key(rootedPath, rootedPath, excludes);
    var result = evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    return ((DirectoryTreeDigestValue) result.get(key)).hexDigest();
  }

  @Test
  public void basic() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    String oldDigest = getTreeDigest("/");

    scratch.overwriteFile("b/b", "something else");
    assertThat(getTreeDigest("/")).isNotEqualTo(oldDigest);
  }

  @Test
  public void basicExcludes() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    String excludeB = "**/b/b";
    String oldDigest = getTreeDigest("/", ImmutableList.of(excludeB));

    scratch.overwriteFile("b/b", "something else");
    assertThat(getTreeDigest("/", ImmutableList.of(excludeB))).isEqualTo(oldDigest);
  }

  @Test
  public void addFile() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    String oldDigest = getTreeDigest("/");

    scratch.file("b/d", "something else");
    String updatedDigest = getTreeDigest("/");
    assertThat(updatedDigest).isNotEqualTo(oldDigest);

    scratch.file("b/ignoredFile", "ignored");
    assertThat(getTreeDigest("/", ImmutableList.of("**/ignoredFile"))).isEqualTo(updatedDigest);
  }

  @Test
  public void removeFile() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    scratch.file("ignoredFile", "ignored");
    String ignorePattern = "**/ignoredFile";
    String oldDigest = getTreeDigest("/", ImmutableList.of(ignorePattern));

    scratch.deleteFile("ignoredFile");
    assertThat(getTreeDigest("/", ImmutableList.of(ignorePattern))).isEqualTo(oldDigest);

    scratch.deleteFile("b/b");
    assertThat(getTreeDigest("/", ImmutableList.of(ignorePattern))).isNotEqualTo(oldDigest);
  }

  @Test
  public void renameFile() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    scratch.file("ignoredFile", "ignored");
    String ignorePattern = "**/ignored*";
    String oldDigest = getTreeDigest("/", ImmutableList.of(ignorePattern));

    scratch.deleteFile("ignoredFile");
    scratch.file("ignoredFileRenamed", "ignored");
    assertThat(getTreeDigest("/", ImmutableList.of(ignorePattern))).isEqualTo(oldDigest);

    scratch.deleteFile("b/b");
    scratch.file("b/b1", "b");
    assertThat(getTreeDigest("/", ImmutableList.of(ignorePattern))).isNotEqualTo(oldDigest);
  }

  @Test
  public void swapDirAndFile() throws Exception {
    scratch.file("a", "a");
    scratch.file("b", "b");
    scratch.file("c/inner", "inner");
    String oldDigest = getTreeDigest("/");

    scratch.resolve("c").deleteTree();
    scratch.deleteFile("b");
    scratch.file("b/inner", "inner");
    scratch.file("c", "b");
    assertThat(getTreeDigest("/")).isNotEqualTo(oldDigest);
  }

  @Test
  public void changeMtime() throws Exception {
    scratch.file("a", "a");
    scratch.file("b", "b");
    scratch.file("c", "c");

    String oldDigest = getTreeDigest("/");

    // We don't digest mtimes so this shouldn't affect anything.
    scratch.resolve("c").setLastModifiedTime(2024L);
    assertThat(getTreeDigest("/")).isEqualTo(oldDigest);
  }

  @Test
  public void symlink() throws Exception {
    scratch.file("dir/a", "a");
    scratch.resolve("dir/b").createSymbolicLink(scratch.resolve("otherdir"));
    scratch.file("dir/c", "c");
    scratch.file("otherdir/b", "b");
    scratch.file("otherdir/sub/sub", "sub");
    String oldDigest = getTreeDigest("dir");

    scratch.deleteFile("dir/b");
    scratch.resolve("dir/b").createSymbolicLink(scratch.resolve("yetotherdir"));
    scratch.file("yetotherdir/crazy", "stuff");
    assertThat(getTreeDigest("dir")).isNotEqualTo(oldDigest);
  }

  @Test
  public void symlinkRetargetedToDirectoryThatIsAlsoAnEntry() throws Exception {
    scratch.file("dir/a/data", "X");
    scratch.file("dir/b/data", "Y");
    scratch.resolve("dir/c").createSymbolicLink(scratch.resolve("dir/a"));
    // Like a watched tree in the workspace, the directory lies under a package root, so that the
    // entry a and the symlink c resolve to the same rooted path.
    RootedPath dir =
        RootedPath.toRootedPath(Root.fromPath(scratch.resolve("")), PathFragment.create("dir"));
    String oldDigest = getTreeDigest(dir, ImmutableList.of());

    // The entries resolve to the same set of directories as before, but c now has b's contents.
    scratch.deleteFile("dir/c");
    scratch.resolve("dir/c").createSymbolicLink(scratch.resolve("dir/b"));
    assertThat(getTreeDigest(dir, ImmutableList.of())).isNotEqualTo(oldDigest);
  }

  @Test
  public void symlinkToExcludedDirectory_matchedUnderItsOwnName() throws Exception {
    scratch.file("dir/ignored/data", "X");
    scratch.resolve("dir/keep").createSymbolicLink(scratch.resolve("dir/ignored"));
    ImmutableList<String> excludes = ImmutableList.of("ignored/**");
    // Like a watched tree in the workspace, the directory lies under a package root, so that the
    // real path of the symlink's target is matched against the excludes.
    RootedPath dir =
        RootedPath.toRootedPath(Root.fromPath(scratch.resolve("")), PathFragment.create("dir"));
    String oldDigest = getTreeDigest(dir, excludes);

    // keep/data is part of the tree even though the directory it resolves to is excluded.
    scratch.overwriteFile("dir/ignored/data", ImmutableList.of("Y"));
    assertThat(getTreeDigest(dir, excludes)).isNotEqualTo(oldDigest);
  }

  @Test
  public void danglingSymlink() throws Exception {
    scratch.file("dir/a", "a");
    scratch.resolve("dir/b").createSymbolicLink(scratch.resolve("otherdir"));
    scratch.file("dir/c", "c");
    String oldDigest = getTreeDigest("dir");

    scratch.file("otherdir/b", "b");
    assertThat(getTreeDigest("dir")).isNotEqualTo(oldDigest);
  }

  @Test
  public void symlinkPointingToSameContents() throws Exception {
    scratch.file("dir/a", "a");
    scratch.file("dir/b/b", "b");
    scratch.file("dir/b/sub/sub", "sub");
    scratch.file("dir/c", "c");
    String oldDigest = getTreeDigest("dir");

    // replace dir/b with a symlink pointing to otherdir/, which contains the same contents.
    // this shouldn't affect the tree digest.
    scratch.resolve("dir/b").deleteTree();
    scratch.resolve("dir/b").createSymbolicLink(scratch.resolve("otherdir"));
    scratch.file("otherdir/b", "b");
    scratch.file("otherdir/sub/sub", "sub");
    assertThat(getTreeDigest("dir")).isEqualTo(oldDigest);
  }

  public static boolean excludes(DirectoryTreeDigestValue.Key key, String path) {
    return DirectoryTreeDigestFunction.excludes(path, key.globBase(), key.excludes(), null);
  }

  public static boolean excludes(DirectoryTreeDigestValue.Key key, RootedPath path) {
    return DirectoryTreeDigestFunction.excludes(path, key.globBase(), key.excludes(), null);
  }

  @Test
  public void symlinkToAncestor_infiniteSymlinkExpansion() throws Exception {
    // The infinite expansion is reported as an error event.
    reporter.removeHandler(failFastHandler);
    scratch.file("dir/a", "a");
    scratch.resolve("dir/loop").createSymbolicLink(scratch.resolve("dir"));

    assertThrows(FileSymlinkInfiniteExpansionException.class, () -> getTreeDigest("dir"));
    // Also with excludes, which make the digest of a directory depend on the path it is reached
    // through.
    assertThrows(
        FileSymlinkInfiniteExpansionException.class,
        () -> getTreeDigest("dir", ImmutableList.of("unrelated")));
    // The symlink isn't followed if it is excluded.
    var oldDigest = getTreeDigest("dir", ImmutableList.of("loop"));
    scratch.overwriteFile("dir/a", "b");
    assertThat(getTreeDigest("dir", ImmutableList.of("loop"))).isNotEqualTo(oldDigest);
  }

  @Test
  public void symlinksToSameDirectory_singleDigest() throws Exception {
    scratch.file("dir/real/sub/data", "X");
    scratch.resolve("dir/alias1").createSymbolicLink(scratch.resolve("dir/real"));
    scratch.resolve("dir/alias2").createSymbolicLink(scratch.resolve("dir/real"));
    RootedPath dir =
        RootedPath.toRootedPath(Root.fromPath(scratch.resolve("")), PathFragment.create("dir"));
    var evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);

    getTreeDigest(evaluator, dir, ImmutableList.of());

    // dir, real and real/sub, regardless of the three paths under which real is reached.
    assertThat(
            evaluator.getDoneValues().keySet().stream()
                .filter(key -> key.functionName().equals(SkyFunctions.DIRECTORY_TREE_DIGEST))
                .count())
        .isEqualTo(3);
  }

  @Test
  public void keyBasicExcludes() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));
    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(
            rootedPath, rootedPath, ImmutableList.of("ignoredFile", "**/*.tmp"));

    assertThat(excludes(key, "foo/bar/ignoredFile")).isTrue();
    assertThat(excludes(key, "foo/bar/anything.ending.in.tmp")).isTrue();
    assertThat(excludes(key, "foo/bar/anything/ending/in/file.tmp")).isTrue();
    assertThat(excludes(key, "foo/bar/notIgnored")).isFalse();
  }

  @Test
  public void keyDifferentRoots() {
    Path pkg1 = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg1), PathFragment.create("foo/bar"));

    Path pkg2 = root.getRelative("pkg2");
    RootedPath differentRoot =
        RootedPath.toRootedPath(Root.fromPath(pkg2), PathFragment.create("foo/bar/ignoredFile"));

    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(rootedPath, rootedPath, ImmutableList.of("ignoredFile"));

    assertThat(excludes(key, differentRoot)).isFalse();
  }

  @Test
  public void keySameRoots() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));
    RootedPath sameRootIgnoredFile =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar/ignoredFile"));

    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(rootedPath, rootedPath, ImmutableList.of("ignoredFile"));
    assertThat(excludes(key, sameRootIgnoredFile)).isTrue();
  }

  @Test
  public void keyEmptyExcludes() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));

    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(rootedPath, rootedPath, ImmutableList.of());
    assertThat(excludes(key, "/pkg/foo/bar")).isFalse();
  }
}
