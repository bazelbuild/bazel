// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.RootedPathAndCasing;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PathCasingLookupFunction}. */
@RunWith(JUnit4.class)
public final class PathCasingLookupFunctionTest extends FoundationTestCase {

  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;

  @Before
  public final void setUp() {
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, rootDirectory),
            rootDirectory,
            null,
            AnalysisMock.get().getProductName());
    ExternalFilesHelper externalFilesHelper =
        ExternalFilesHelper.createForTesting(
            pkgLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);

    AtomicReference<UnixGlob.FilesystemCalls> syscalls =
        new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS);
    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(
        FileStateValue.FILE_STATE,
        new FileStateFunction(new AtomicReference<>(), syscalls, externalFilesHelper));
    skyFunctions.put(FileValue.FILE, new FileFunction(pkgLocator));
    skyFunctions.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    skyFunctions.put(
        SkyFunctions.DIRECTORY_LISTING_STATE,
        new DirectoryListingStateFunction(externalFilesHelper, syscalls));
    skyFunctions.put(SkyFunctions.PATH_CASING_LOOKUP, new PathCasingLookupFunction());

    differencer = new SequencedRecordingDifferencer();
    driver =
        new SequentialBuildDriver(new InMemoryMemoizingEvaluator(skyFunctions, differencer, null));
  }

  private RootedPath rootedPath(String relative) {
    return RootedPath.toRootedPath(Root.fromPath(rootDirectory), PathFragment.create(relative));
  }

  @Test
  public void testSanityCheckFilesystemIsCaseInsensitive() {
    Path p1 = rootDirectory.getRelative("Foo/Bar");
    Path p2 = rootDirectory.getRelative("FOO/BAR");
    Path p3 = rootDirectory.getRelative("control");
    assertThat(p1).isNotSameInstanceAs(p2);
    assertThat(p1).isNotSameInstanceAs(p3);
    assertThat(p2).isNotSameInstanceAs(p3);
    assertThat(p1).isEqualTo(p2);
    assertThat(p1).isNotEqualTo(p3);
  }

  @Test
  public void testPathCasingLookup() throws Exception {
    RootedPath a = rootedPath("Foo/Bar/Baz");
    RootedPath b = rootedPath("fOO/baR/BAZ");
    createFile(a);
    assertThat(a).isEqualTo(b);
    assertThat(RootedPathAndCasing.create(a)).isNotEqualTo(RootedPathAndCasing.create(b));
    assertThat(expectEvalSuccess(a).isCorrect()).isTrue();
    assertThat(expectEvalSuccess(b).isCorrect()).isFalse();
  }

  @Test
  public void testNonExistentPath() throws Exception {
    RootedPath file = rootedPath("Foo/Bar/Baz.txt");
    createFile(file);
    RootedPath missing1 = rootedPath("Foo/Bar/x/y");
    RootedPath missing2 = rootedPath("Foo/BAR/x/y");
    // Non-existent paths are correct if their existing part is correct.
    assertThat(expectEvalSuccess(missing1).isCorrect()).isTrue();
    assertThat(expectEvalSuccess(missing2).isCorrect()).isFalse();
    // Non-existent paths are illegal if their parent exists but is not a directory.
    RootedPath bad = rootedPath("Foo/Bar/Baz.txt/x/y");
    Exception e = expectEvalFailure(bad);
    assertThat(e).hasMessageThat().contains("its parent exists but is not a directory");
  }

  @Test
  public void testNonExistentPathThatComesIntoExistence() throws Exception {
    RootedPath a = rootedPath("Foo/Bar/Baz");
    RootedPath b = rootedPath("fOO/baR/BAZ");
    assertThat(a).isEqualTo(b);
    // Expecting RootedPath.toRootedPath not to intern instances, otherwise 'a' would be the same
    // instance as 'b' which would nullify this test.
    assertThat(a).isNotSameInstanceAs(b);
    assertThat(a.toString()).isNotEqualTo(b.toString());
    assertThat(RootedPathAndCasing.create(a)).isNotEqualTo(RootedPathAndCasing.create(b));
    // Path does not exist, so both casings are correct!
    assertThat(expectEvalSuccess(a).isCorrect()).isTrue();
    assertThat(expectEvalSuccess(b).isCorrect()).isTrue();
    // Path comes into existence.
    createFile(a);
    // Now only one casing is correct.
    assertThat(expectEvalSuccess(a).isCorrect()).isTrue();
    assertThat(expectEvalSuccess(b).isCorrect()).isFalse();
  }

  @Test
  public void testExistingPathThatIsThenDeleted() throws Exception {
    RootedPath a = rootedPath("Foo/Bar/Baz");
    RootedPath b = rootedPath("Foo/Bar/BAZ");
    createFile(a);
    // Path exists, so only one casing is correct.
    assertThat(expectEvalSuccess(a).isCorrect()).isTrue();
    assertThat(expectEvalSuccess(b).isCorrect()).isFalse();
    // Path no longer exists, both casings are correct.
    deleteFile(a);
    assertThat(expectEvalSuccess(a).isCorrect()).isTrue();
    assertThat(expectEvalSuccess(b).isCorrect()).isTrue();
  }

  private void createFile(RootedPath p) throws IOException {
    Path path = p.asPath();
    if (!path.getParentDirectory().exists()) {
      scratch.dir(path.getParentDirectory().getPathString());
    }
    scratch.file(path.getPathString());
    invalidateFileAndParents(p);
  }

  private void deleteFile(RootedPath p) throws IOException {
    Path path = p.asPath();
    scratch.deleteFile(path.getPathString());
    invalidateFileAndParents(p);
  }

  private EvaluationResult<PathCasingLookupValue> evaluate(SkyKey key) throws Exception {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(NullEventHandler.INSTANCE)
            .build();
    return driver.evaluate(ImmutableList.of(key), evaluationContext);
  }

  private PathCasingLookupValue expectEvalSuccess(RootedPath path) throws Exception {
    SkyKey key = PathCasingLookupValue.key(path);
    EvaluationResult<PathCasingLookupValue> result = evaluate(key);
    assertThat(result.hasError()).isFalse();
    return result.get(key);
  }

  private Exception expectEvalFailure(RootedPath path) throws Exception {
    SkyKey key = PathCasingLookupValue.key(path);
    EvaluationResult<PathCasingLookupValue> result = evaluate(key);
    assertThat(result.hasError()).isTrue();
    return result.getError().getException();
  }

  private void invalidateFile(RootedPath path) {
    differencer.invalidate(ImmutableList.of(FileStateValue.key(path)));
  }

  private void invalidateDirectory(RootedPath path) {
    invalidateFile(path);
    differencer.invalidate(ImmutableList.of(DirectoryListingStateValue.key(path)));
  }

  private void invalidateFileAndParents(RootedPath p) {
    invalidateFile(p);
    do {
      p = p.getParentDirectory();
      invalidateDirectory(p);
    } while (!p.getRootRelativePath().isEmpty());
  }
}
