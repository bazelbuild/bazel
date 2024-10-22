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

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStateKey;
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
  public void setup() throws Exception {
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
            /* defaultSystemJavabase= */ null,
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
            .buildOrThrow();

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, packageLocator.get());
  }

  private String getTreeDigest(String path) throws Exception {
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.absoluteRoot(fileSystem), scratch.resolve(path));
    SkyKey key = DirectoryTreeDigestValue.key(rootedPath);
    MemoizingEvaluator evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
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
  public void addFile() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    String oldDigest = getTreeDigest("/");

    scratch.file("b/d", "something else");
    assertThat(getTreeDigest("/")).isNotEqualTo(oldDigest);
  }

  @Test
  public void removeFile() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    String oldDigest = getTreeDigest("/");

    scratch.deleteFile("b/b");
    assertThat(getTreeDigest("/")).isNotEqualTo(oldDigest);
  }

  @Test
  public void renameFile() throws Exception {
    scratch.file("a", "a");
    scratch.file("b/b", "b");
    scratch.file("c", "c");
    String oldDigest = getTreeDigest("/");

    scratch.deleteFile("b/b");
    scratch.file("b/b1", "b");
    assertThat(getTreeDigest("/")).isNotEqualTo(oldDigest);
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
}
