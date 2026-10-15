// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Reason;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/**
 * Tests for {@link SymlinkTreeStrategy} on a file system that doesn't support symlinks natively
 * (i.e. Windows without {@code --windows_enable_symlinks}), where symlinks to files are emulated by
 * copies and thus require the file to exist.
 */
@RunWith(JUnit4.class)
public final class SymlinkTreeStrategyPrefetchTest extends BuildViewTestCase {

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256) {
      @Override
      public boolean supportsSymbolicLinksNatively(PathFragment path) {
        return false;
      }
    };
  }

  @Test
  public void prefetchesRegularFileRunfilesBeforeCreatingTree() throws Exception {
    ActionExecutionContext context = mock(ActionExecutionContext.class);
    OutputService outputService = mock(OutputService.class);
    ActionInputPrefetcher prefetcher = mock(ActionInputPrefetcher.class);
    StaticInputMetadataProvider metadataProvider = StaticInputMetadataProvider.empty();

    when(context.getContext(SymlinkTreeActionContext.class))
        .thenReturn(new SymlinkTreeStrategy(outputService, TestConstants.WORKSPACE_NAME));
    when(context.getInputPath(any())).thenAnswer((i) -> ((Artifact) i.getArgument(0)).getPath());
    when(context.getEventHandler()).thenReturn(new StoredEventHandler());
    when(context.getActionInputPrefetcher()).thenReturn(prefetcher);
    when(context.getInputMetadataProvider()).thenReturn(metadataProvider);
    when(outputService.canCreateSymlinkTree()).thenReturn(false);
    when(prefetcher.prefetchFiles(any(), any(), any(), any(), any(), any()))
        .thenReturn(immediateVoidFuture());

    Artifact inputManifest = getBinArtifactWithNoOwner("dir/manifest.in");
    Artifact outputManifest = getBinArtifactWithNoOwner("dir.runfiles/MANIFEST");
    Artifact file = getBinArtifactWithNoOwner("dir/file");
    ArtifactRoot root = file.getRoot();
    SpecialArtifact tree = ActionsTestUtil.createTreeArtifactWithGeneratingAction(root, "dir/tree");
    SpecialArtifact symlink = ActionsTestUtil.createUnresolvedSymlinkArtifact(root, "dir/symlink");
    FileSystemUtils.ensureSymbolicLink(symlink.getPath(), "/path/to/target");

    Runfiles runfiles =
        new Runfiles.Builder("TESTING")
            .addArtifact(file)
            .addArtifact(tree)
            .addArtifact(symlink)
            .build();
    SymlinkTreeAction action =
        new SymlinkTreeAction(
            ActionsTestUtil.NULL_ACTION_OWNER,
            inputManifest,
            runfiles,
            outputManifest,
            /* repoMappingManifest= */ null,
            ActionEnvironment.EMPTY,
            RunfileSymlinksMode.CREATE,
            "workspace");

    action.execute(context);

    // Only regular files have to be present on disk: tree artifacts are linked via junctions and
    // unresolved symlinks are created textually.
    @SuppressWarnings("unchecked")
    ArgumentCaptor<Supplier<Iterable<? extends ActionInput>>> inputsCaptor =
        ArgumentCaptor.forClass(Supplier.class);
    verify(prefetcher)
        .prefetchFiles(
            eq(action),
            isNull(),
            inputsCaptor.capture(),
            eq(metadataProvider),
            eq(Priority.CRITICAL),
            eq(Reason.INPUTS));
    assertThat(inputsCaptor.getValue().get()).containsExactly(file);

    Path treeRoot = outputManifest.getPath().getParentDirectory().getRelative("TESTING/dir");
    assertThat(treeRoot.getRelative("file").readSymbolicLink())
        .isEqualTo(file.getPath().asFragment());
    assertThat(treeRoot.getRelative("tree").readSymbolicLink())
        .isEqualTo(tree.getPath().asFragment());
    assertThat(treeRoot.getRelative("symlink").readSymbolicLink())
        .isEqualTo(PathFragment.create("/path/to/target"));
    assertThat(treeRoot.getRelative("symlink").exists(Symlinks.NOFOLLOW)).isTrue();
  }
}
