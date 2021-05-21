// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testing.common.DirectoryListingHelper.leafDirectoryEntries;
import static org.junit.Assert.assertThrows;

import com.google.common.cache.CacheBuilderSpec;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.skyframe.ActionOutputDirectoryHelper.CreateOutputDirectoryException;
import com.google.devtools.build.lib.testing.common.DirectoryListingHelper;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DelegateFileSystem;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ActionOutputDirectoryHelper}. */
@RunWith(TestParameterInjector.class)
public class ActionOutputDirectoryHelperTest {

  private Path execRoot;
  private ArtifactRoot outputRoot;

  @Before
  public void createArtifactRootAndOutputDirectoryHelper() throws IOException {
    Scratch scratch = new Scratch();
    execRoot = scratch.dir("/execroot");
    outputRoot = createOutputRoot(execRoot);
  }

  enum DirectoryCache {
    CACHE_ENABLED(CacheBuilderSpec.parse("maximumSize=100000")),
    CACHE_DISABLED(CacheBuilderSpec.disableCaching());

    @SuppressWarnings("ImmutableEnumChecker")
    final CacheBuilderSpec spec;

    DirectoryCache(CacheBuilderSpec spec) {
      this.spec = spec;
    }
  }

  private enum OutputSet {
    SINGLE_FILE(
        /*fileOutputs=*/ ImmutableSet.of("a/b"),
        /*treeOutputs=*/ ImmutableSet.of(),
        /*expectedDirectories=*/ ImmutableList.of("a")),
    DEEP_DIRECTORY_STRUCTURE(
        /*fileOutputs=*/ ImmutableSet.of("a/b/c/d/e/f/g/h/i/j/k/l/m"),
        /*treeOutputs=*/ ImmutableSet.of(),
        /*expectedDirectories=*/ ImmutableList.of("a/b/c/d/e/f/g/h/i/j/k/l")),
    MULTIPLE_FILES(
        /*fileOutputs=*/ ImmutableSet.of("a/b/c", "a/c", "a/d/1", "a/d/2"),
        /*treeOutputs=*/ ImmutableSet.of(),
        /*expectedDirectories=*/ ImmutableList.of("a/b", "a/d")),
    TREE_OUTPUT(
        /*fileOutputs=*/ ImmutableSet.of(),
        /*treeOutputs=*/ ImmutableSet.of("a/b"),
        /*expectedDirectories=*/ ImmutableList.of("a/b"));

    OutputSet(
        ImmutableSet<String> fileOutputs,
        ImmutableSet<String> treeOutputs,
        ImmutableList<String> expectedDirectories) {
      this.fileOutputs = fileOutputs;
      this.treeOutputs = treeOutputs;
      this.expectedDirectories = expectedDirectories;
    }

    ImmutableSet<Artifact> actionOutputs(ActionOutputDirectoryHelperTest test) {
      ImmutableSet.Builder<Artifact> outs =
          ImmutableSet.builderWithExpectedSize(fileOutputs.size() + treeOutputs.size());
      fileOutputs.stream().map(test::createOutput).forEach(outs::add);
      treeOutputs.stream().map(test::createTreeOutput).forEach(outs::add);
      return outs.build();
    }

    ImmutableList<Dirent> expectedDirectoryEntries() {
      return expectedDirectories.stream()
          .map(DirectoryListingHelper::directory)
          .collect(toImmutableList());
    }

    private final ImmutableSet<String> fileOutputs;
    private final ImmutableSet<String> treeOutputs;
    private final ImmutableList<String> expectedDirectories;
  }

  @Test
  public void createOutputDirectories_createsExpectedDirectories(
      @TestParameter DirectoryCache cache, @TestParameter OutputSet outputSet) throws Exception {
    ActionOutputDirectoryHelper outputDirectoryHelper = new ActionOutputDirectoryHelper(cache.spec);

    outputDirectoryHelper.createOutputDirectories(outputSet.actionOutputs(this));

    assertThat(leafDirectoryEntries(outputRoot.getRoot().asPath()))
        .containsExactlyElementsIn(outputSet.expectedDirectoryEntries());
  }

  @Test
  public void createActionFsOutputDirectories_createsExpectedDirectoriesInActionFs(
      @TestParameter OutputSet outputSet) throws Exception {
    ActionOutputDirectoryHelper outputDirectoryHelper = createActionOutputDirectoryHelper();
    FileSystem actionFileSystem = new Scratch().getFileSystem();
    ArtifactPathResolver resolver =
        ArtifactPathResolver.createPathResolver(actionFileSystem, execRoot);

    outputDirectoryHelper.createActionFsOutputDirectories(outputSet.actionOutputs(this), resolver);

    Path outputRootPath = outputRoot.getRoot().asPath();
    assertThat(outputRootPath.exists()).isFalse();
    assertThat(leafDirectoryEntries(actionFileSystem.getPath(outputRootPath.asFragment())))
        .containsExactlyElementsIn(outputSet.expectedDirectoryEntries());
  }

  @Test
  public void createOutputDirectories_ioExceptionWhenCreatingDirectory_fails(
      @TestParameter DirectoryCache cache) {
    ActionOutputDirectoryHelper outputDirectoryHelper = new ActionOutputDirectoryHelper(cache.spec);
    IOException injectedException = new IOException("oh no!");
    PathFragment outputRootPath = outputRoot.getRoot().asPath().asFragment();
    FileSystem fsWithFailures =
        createFileSystemInjectingException(outputRootPath.getRelative("dir"), injectedException);
    ArtifactRoot rootWithFailure = createOutputRoot(fsWithFailures.getPath(execRoot.asFragment()));
    ImmutableSet<Artifact> outputs = ImmutableSet.of(createOutput(rootWithFailure, "dir/file"));

    CreateOutputDirectoryException e =
        assertThrows(
            CreateOutputDirectoryException.class,
            () -> outputDirectoryHelper.createOutputDirectories(outputs));

    assertThat(e.directoryPath).isEqualTo(outputRootPath.getRelative("dir"));
    assertThat(e).hasCauseThat().isSameInstanceAs(injectedException);
  }

  @Test
  public void createActionFsOutputDirectories_ioExceptionWhenCreatingDirectory_fails() {
    ActionOutputDirectoryHelper outputDirectoryHelper = createActionOutputDirectoryHelper();
    IOException injectedException = new IOException("oh no!");
    PathFragment outputRootPath = outputRoot.getRoot().asPath().asFragment();
    FileSystem fsWithFailures =
        createFileSystemInjectingException(outputRootPath.getRelative("dir"), injectedException);
    ArtifactRoot rootWithFailure = createOutputRoot(fsWithFailures.getPath(execRoot.asFragment()));
    ImmutableSet<Artifact> outputs = ImmutableSet.of(createOutput(rootWithFailure, "dir/file"));
    FileSystem actionFileSystem = new DelegateFileSystem(fsWithFailures) {};
    ArtifactPathResolver pathResolver =
        ArtifactPathResolver.createPathResolver(actionFileSystem, execRoot);

    CreateOutputDirectoryException e =
        assertThrows(
            CreateOutputDirectoryException.class,
            () -> outputDirectoryHelper.createActionFsOutputDirectories(outputs, pathResolver));

    assertThat(e.directoryPath).isEqualTo(outputRootPath.getRelative("dir"));
    assertThat(e).hasCauseThat().isSameInstanceAs(injectedException);
  }

  private FileSystem createFileSystemInjectingException(
      PathFragment failingPath, IOException injectedException) {
    return new DelegateFileSystem(execRoot.getFileSystem()) {
      @Override
      public boolean createDirectory(PathFragment path) throws IOException {
        if (path.equals(failingPath)) {
          throw injectedException;
        }
        return super.createDirectory(path);
      }

      @Override
      public void createDirectoryAndParents(PathFragment path) throws IOException {
        if (path.equals(failingPath)) {
          throw injectedException;
        }
        super.createDirectoryAndParents(path);
      }
    };
  }

  private SpecialArtifact createTreeOutput(String relativeExecPath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        outputRoot, outputRoot.getExecPath().getRelative(relativeExecPath));
  }

  private Artifact createOutput(String relativeExecPath) {
    return createOutput(outputRoot, relativeExecPath);
  }

  private static Artifact createOutput(ArtifactRoot outputRoot, String relativeExecPath) {
    return ActionsTestUtil.createArtifactWithRootRelativePath(
        outputRoot, PathFragment.create(relativeExecPath));
  }

  private static ArtifactRoot createOutputRoot(Path execRoot) {
    return ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "out");
  }

  private static ActionOutputDirectoryHelper createActionOutputDirectoryHelper() {
    return new ActionOutputDirectoryHelper(DirectoryCache.CACHE_ENABLED.spec);
  }
}
