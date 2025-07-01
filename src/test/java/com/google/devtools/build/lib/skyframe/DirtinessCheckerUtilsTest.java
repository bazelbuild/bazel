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
//

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.UnionDirtinessChecker;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.EnumSet;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mockito;

/** Tests for {@link DirtinessCheckerUtils}. */
@RunWith(TestParameterInjector.class)
public final class DirtinessCheckerUtilsTest {
  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path pkgRoot = fs.getPath("/testroot");
  private final Root srcRoot = Root.fromPath(pkgRoot);
  private final Path outputBase = fs.getPath("/outputroot/user/outputBase");
  private final AtomicReference<PathPackageLocator> pkgLocator =
      new AtomicReference<>(
          new PathPackageLocator(
              outputBase,
              ImmutableList.of(srcRoot),
              BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
  private final BlazeDirectories directories =
      new BlazeDirectories(
          new ServerDirectories(pkgRoot, outputBase, outputBase.getParentDirectory()),
          pkgRoot,
          /* defaultSystemJavabase= */ null,
          TestConstants.PRODUCT_NAME);
  private final ExternalFilesHelper externalFilesHelper =
      ExternalFilesHelper.createForTesting(
          pkgLocator,
          ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
          directories);

  @Test
  public void missingDiffChecker_matchesInsideRoot() {
    assertThat(
            createMissingDiffChecker()
                .applies(RootedPath.toRootedPath(srcRoot, PathFragment.create("bar"))))
        .isTrue();
  }

  @Test
  public void missingDiffChecker_doesntMatchIfRootDoesntMatch() {
    assertThat(
            createMissingDiffChecker()
                .applies(RootedPath.toRootedPath(Root.absoluteRoot(fs), pkgRoot.asFragment())))
        .isFalse();
  }

  @Test
  public void check_usesSyscallCache_andReturnsNewValue(
      @TestParameter boolean externalChecker, @TestParameter boolean internalFile)
      throws IOException {
    SyscallCache spyCache = spy(SyscallCache.NO_CACHE);
    RootedPath rootedPath = internalFile ? makeInternalRootedPath() : makeExternalRootedPath();
    Path path = rootedPath.asPath();
    SkyValueDirtinessChecker underTest =
        externalChecker
            ? new DirtinessCheckerUtils.ExternalDirtinessChecker(
                externalFilesHelper,
                EnumSet.of(
                    ExternalFilesHelper.FileType.INTERNAL,
                    ExternalFilesHelper.FileType.EXTERNAL_OTHER))
            : createMissingDiffChecker();

    boolean shouldCheck = underTest.applies(rootedPath);
    assertThat(shouldCheck).isEqualTo(externalChecker || internalFile);

    Assume.assumeTrue("Missing diff checker doesn't apply to external files", shouldCheck);

    assertThat(underTest.check(rootedPath, null, /* oldMtsv= */ null, spyCache, null))
        .isEqualTo(
            SkyValueDirtinessChecker.DirtyResult.dirtyWithNewValue(
                FileStateValue.NONEXISTENT_FILE_STATE_NODE));

    verify(spyCache).getType(path, Symlinks.NOFOLLOW);
    verify(spyCache).statIfFound(path, Symlinks.NOFOLLOW);
    verifyNoMoreInteractions(spyCache);
  }

  private RootedPath makeInternalRootedPath() {
    return RootedPath.toRootedPath(srcRoot, PathFragment.create("srcfile"));
  }

  private RootedPath makeExternalRootedPath() {
    return RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/extfile"));
  }

  @Test
  public void skipsSyscallCacheForRepoFile_andDoesntReturnNewValue(
      @TestParameter boolean externalChecker) throws Exception {
    ExternalFilesHelper externalFilesHelper = this.externalFilesHelper;
    RootedPath rootedPath =
        RootedPath.toRootedPath(
            Root.fromPath(outputBase),
            LabelConstants.EXTERNAL_REPOSITORY_LOCATION.getChild("extrepofile"));
    SkyValueDirtinessChecker underTest =
        externalChecker
            ? new DirtinessCheckerUtils.ExternalDirtinessChecker(
                externalFilesHelper, EnumSet.of(ExternalFilesHelper.FileType.EXTERNAL_REPO))
            : new DirtinessCheckerUtils.MissingDiffDirtinessChecker(ImmutableSet.of(srcRoot));

    boolean shouldCheck = underTest.applies(rootedPath);
    assertThat(shouldCheck).isEqualTo(externalChecker);

    Assume.assumeTrue("Missing diff checker doesn't apply to external files", shouldCheck);

    SyscallCache mockCache = mock(SyscallCache.class);

    assertThat(underTest.check(rootedPath, null, /* oldMtsv= */ null, mockCache, null))
        .isEqualTo(SkyValueDirtinessChecker.DirtyResult.dirty());

    Mockito.verifyNoInteractions(mockCache);
  }

  @Test
  public void externalDiffChecker_doesntMatchType() {
    DirtinessCheckerUtils.ExternalDirtinessChecker underTest =
        new DirtinessCheckerUtils.ExternalDirtinessChecker(
            externalFilesHelper, EnumSet.of(ExternalFilesHelper.FileType.EXTERNAL_REPO));

    assertThat(
            underTest.applies(
                RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/file"))))
        .isFalse();
  }

  @Test
  public void missingDiffDirtinessCheckers_nullMaxTransitiveSourceVersionForNewValue()
      throws Exception {
    SkyKey key = mock(SkyKey.class);
    SkyValue value = mock(SkyValue.class);
    DirtinessCheckerUtils.MissingDiffDirtinessChecker underTest = createMissingDiffChecker();

    assertThat(underTest.getMaxTransitiveSourceVersionForNewValue(key, value)).isNull();
  }

  @Test
  public void externalDirtinessCheckers_nullMaxTransitiveSourceVersionForNewValue()
      throws Exception {
    SkyKey key = mock(SkyKey.class);
    SkyValue value = mock(SkyValue.class);
    DirtinessCheckerUtils.ExternalDirtinessChecker underTest =
        new DirtinessCheckerUtils.ExternalDirtinessChecker(
            externalFilesHelper, EnumSet.of(ExternalFilesHelper.FileType.EXTERNAL_REPO));

    assertThat(underTest.getMaxTransitiveSourceVersionForNewValue(key, value)).isNull();
  }

  @Test
  public void unionDirtinessChecker_nullMaxTransitiveSourceVersionForNewValue() throws Exception {
    RootedPath rootedPath = makeInternalRootedPath();
    SkyKey key = FileStateValue.key(rootedPath);
    SkyValue value = FileStateValue.create(rootedPath, SyscallCache.NO_CACHE, /* tsgm= */ null);
    UnionDirtinessChecker underTest =
        new UnionDirtinessChecker(
            ImmutableList.of(
                createMissingDiffChecker(),
                new DirtinessCheckerUtils.ExternalDirtinessChecker(
                    externalFilesHelper, EnumSet.of(ExternalFilesHelper.FileType.EXTERNAL_REPO))));

    assertThat(underTest.getMaxTransitiveSourceVersionForNewValue(key, value)).isNull();
  }

  private DirtinessCheckerUtils.MissingDiffDirtinessChecker createMissingDiffChecker() {
    return new DirtinessCheckerUtils.MissingDiffDirtinessChecker(ImmutableSet.of(srcRoot));
  }
}
