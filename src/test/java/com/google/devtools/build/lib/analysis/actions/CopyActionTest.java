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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.OutputMetadataStore;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.io.IOException;
import java.time.Instant;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Tests {@link CopyAction}. */
@RunWith(JUnit4.class)
public class CopyActionTest extends BuildViewTestCase {
  private static final byte[] DIGEST = {1, 2, 3, 4};

  private Executor executor;
  private OutputMetadataStore outputStore;
  private InputMetadataProvider inputProvider;

  @Before
  public void setUp() throws Exception {
    executor = new TestExecutorBuilder(fileSystem, directories).build();
    outputStore = mock(OutputMetadataStore.class);
    inputProvider = mock(InputMetadataProvider.class);
  }

  private void execute(CopyAction action) throws Exception {
    action.execute(
        new ActionExecutionContext(
            executor,
            inputProvider,
            ActionInputPrefetcher.NONE,
            actionKeyContext,
            outputStore,
            /* rewindingEnabled= */ false,
            LostInputsCheck.NONE,
            /* fileOutErr= */ null,
            new StoredEventHandler(),
            /* clientEnv= */ ImmutableMap.of(),
            /* actionFileSystem= */ null,
            DiscoveredModulesPruner.DEFAULT,
            SyscallCache.NO_CACHE,
            ThreadStateReceiver.NULL_INSTANCE));
  }

  private static FileArtifactValue remote(Instant expiry) {
    return FileArtifactValue.createForRemoteFileWithMaterializationData(
        DIGEST, 7, /* locationIndex= */ 3, expiry, /* inMemoryOutput= */ false);
  }

  private SpecialArtifact tree(String name) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        getTargetConfiguration().getBinDir(), name);
  }

  @Test
  public void metadata_isCopyActionWithPassThroughInputs() throws Exception {
    Artifact in = getBinArtifactWithNoOwner("in");
    Artifact out = getBinArtifactWithNoOwner("out");
    CopyAction action = CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying");

    assertThat(action.getMnemonic()).isEqualTo("Copy");
    assertThat(action.mayInsensitivelyPropagateInputs()).isTrue();
    assertThat(action.producesCopyOutputs()).isTrue();
    assertThat(action.getInputs().toList()).containsExactly(in);
    assertThat(action.getOutputs()).containsExactly(out);
    assertThat(action.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isEqualTo(
            CopyAction.create(NULL_ACTION_OWNER, in, out, "Other message")
                .getKey(actionKeyContext, /* inputMetadataProvider= */ null));
  }

  @Test
  public void remoteOnlyFile_injectsFreshRemoteValueAndMovesNoBytes() throws Exception {
    Artifact in = getBinArtifactWithNoOwner("in");
    Artifact out = getBinArtifactWithNoOwner("out");
    FileArtifactValue source = remote(Instant.ofEpochSecond(1000));
    when(inputProvider.getInputMetadata(in)).thenReturn(source);

    execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying"));

    ArgumentCaptor<FileArtifactValue> captor = ArgumentCaptor.forClass(FileArtifactValue.class);
    verify(outputStore).injectFile(eq(out), captor.capture());
    FileArtifactValue injected = captor.getValue();
    assertThat(injected).isNotSameInstanceAs(source);
    assertThat(injected.isRemote()).isTrue();
    assertThat(injected.getDigest()).isEqualTo(DIGEST);
    assertThat(injected.getSize()).isEqualTo(7);
    assertThat(injected.getLocationIndex()).isEqualTo(3);
    assertThat(injected.getExpirationTime()).isEqualTo(Instant.ofEpochSecond(1000));
    assertThat(injected.getResolvedPath()).isNull();
    assertThat(injected.getContentsProxy()).isNull();
    assertThat(injected.isCopyOutput()).isTrue();
    // Mutating one does not affect the other.
    injected.setExpirationTime(Instant.ofEpochSecond(2000));
    assertThat(source.getExpirationTime()).isEqualTo(Instant.ofEpochSecond(1000));
    assertThat(out.getPath().exists()).isFalse();
  }

  @Test
  public void localFile_isCopiedToOwnPathAndNotInjected() throws Exception {
    Artifact in = getBinArtifactWithNoOwner("in");
    Artifact out = getBinArtifactWithNoOwner("sub/out");
    in.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(in.getPath(), UTF_8, "content");
    when(inputProvider.getInputMetadata(in))
        .thenReturn(FileArtifactValue.createForNormalFile(DIGEST, null, 7));

    execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying"));

    assertThat(out.getPath().isSymbolicLink()).isFalse();
    assertThat(FileSystemUtils.readContent(out.getPath(), UTF_8)).isEqualTo("content");
    verify(outputStore, never()).injectFile(any(), any());
  }

  @Test
  public void sourceArtifactWithRemoteLookingMetadata_isStillCopiedLocally() throws Exception {
    Artifact in = getSourceArtifact("in");
    Artifact out = getBinArtifactWithNoOwner("out");
    in.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(in.getPath(), UTF_8, "content");
    when(inputProvider.getInputMetadata(in)).thenReturn(remote(null));

    execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying"));

    assertThat(FileSystemUtils.readContent(out.getPath(), UTF_8)).isEqualTo("content");
    verify(outputStore, never()).injectFile(any(), any());
  }

  @Test
  public void missingLocalSource_failsWithDetailedExitCode() throws Exception {
    Artifact in = getBinArtifactWithNoOwner("in");
    Artifact out = getBinArtifactWithNoOwner("out");
    when(inputProvider.getInputMetadata(in))
        .thenReturn(FileArtifactValue.createForNormalFile(DIGEST, null, 7));

    ActionExecutionException e =
        assertThrows(
            ActionExecutionException.class,
            () -> execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying")));

    assertThat(e).hasMessageThat().contains("failed to copy");
    assertThat(e.getDetailedExitCode().getFailureDetail().getExecution().getCode())
        .isEqualTo(
            Execution.Code.FILE_WRITE_IO_EXCEPTION);
  }

  @Test
  public void fallback_usedWhenPrimaryCopyFails() throws Exception {
    Path from = scratch.file("fallback/from", "hello");
    Path to = scratch.resolve("fallback/deeper/to");
    // The failing primary leaves a partial file behind, which must be replaced.
    CopyAction.copyFileWithFallback(
        from,
        to,
        (f, t) -> {
          FileSystemUtils.writeContent(t, UTF_8, "partial");
          throw new IOException("clone failed");
        });

    assertThat(FileSystemUtils.readContent(to, UTF_8)).isEqualTo("hello\n");
    assertThat(to.isSymbolicLink()).isFalse();
  }

  @Test
  public void fallback_notUsedWhenPrimarySucceeds() throws Exception {
    Path from = scratch.file("fallback/from", "hello");
    Path to = scratch.resolve("fallback/to");
    boolean[] fallbackNeeded = {true};
    CopyAction.copyFileWithFallback(
        from,
        to,
        (f, t) -> {
          fallbackNeeded[0] = false;
          FileSystemUtils.writeContent(t, UTF_8, "primary");
        });

    assertThat(fallbackNeeded[0]).isFalse();
    assertThat(FileSystemUtils.readContent(to, UTF_8)).isEqualTo("primary");
  }

  @Test
  public void fallback_failureLeavesNoPartialOutput() throws Exception {
    Path from = scratch.resolve("fallback/missing");
    Path to = scratch.resolve("fallback/to");

    assertThrows(
        IOException.class,
        () ->
            CopyAction.copyFileWithFallback(
                from,
                to,
                (f, t) -> {
                  FileSystemUtils.writeContent(t, UTF_8, "partial");
                  throw new IOException("boom");
                }));

    assertThat(to.exists()).isFalse();
  }

  @Test
  public void remoteOnlyTree_injectsFreshTreeWithNoResolvedPathOrBytes() throws Exception {
    SpecialArtifact in = tree("in");
    SpecialArtifact out = tree("out");
    TreeFileArtifact a = TreeFileArtifact.createTreeOutput(in, "a");
    TreeFileArtifact b = TreeFileArtifact.createTreeOutput(in, "sub/b");
    FileArtifactValue va = remote(null);
    FileArtifactValue vb = remote(Instant.ofEpochSecond(5));
    when(inputProvider.getTreeMetadata(in))
        .thenReturn(TreeArtifactValue.newBuilder(in).putChild(a, va).putChild(b, vb).build());

    execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying"));

    ArgumentCaptor<TreeArtifactValue> captor = ArgumentCaptor.forClass(TreeArtifactValue.class);
    verify(outputStore).injectTree(eq(out), captor.capture());
    TreeArtifactValue injected = captor.getValue();
    assertThat(injected.getResolvedPath()).isEmpty();
    assertThat(injected.getArchivedRepresentation()).isEmpty();
    TreeFileArtifact outA = TreeFileArtifact.createTreeOutput(out, "a");
    TreeFileArtifact outB = TreeFileArtifact.createTreeOutput(out, "sub/b");
    assertThat(injected.getChildValues()).containsKey(outA);
    assertThat(injected.getChildValues()).containsKey(outB);
    assertThat(injected.getChildValues().get(outA)).isNotSameInstanceAs(va);
    assertThat(injected.getChildValues().get(outB)).isNotSameInstanceAs(vb);
    assertThat(injected.getChildValues().get(outB).getExpirationTime())
        .isEqualTo(Instant.ofEpochSecond(5));
    assertThat(injected.getChildValues().get(outA).getDigest()).isEqualTo(DIGEST);
    assertThat(injected.getChildValues().get(outA).isCopyOutput()).isTrue();
    assertThat(injected.getChildValues().get(outB).isCopyOutput()).isTrue();
    assertThat(out.getPath().exists()).isFalse();
  }

  @Test
  public void mixedTree_copiesLocalChildAndInjectsFreshRemoteChild() throws Exception {
    SpecialArtifact in = tree("in");
    SpecialArtifact out = tree("out");
    TreeFileArtifact local = TreeFileArtifact.createTreeOutput(in, "d/local");
    TreeFileArtifact rem = TreeFileArtifact.createTreeOutput(in, "rem");
    local.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(local.getPath(), UTF_8, "local bytes");
    FileArtifactValue localValue = FileArtifactValue.createForNormalFile(DIGEST, null, 11);
    FileArtifactValue remValue = remote(null);
    when(inputProvider.getTreeMetadata(in))
        .thenReturn(
            TreeArtifactValue.newBuilder(in)
                .putChild(local, localValue)
                .putChild(rem, remValue)
                .build());
    TreeFileArtifact outLocal = TreeFileArtifact.createTreeOutput(out, "d/local");
    FileArtifactValue storeRead = FileArtifactValue.createForNormalFile(DIGEST, null, 11);
    when(outputStore.getTreeArtifactValue(out))
        .thenReturn(TreeArtifactValue.newBuilder(out).putChild(outLocal, storeRead).build());

    execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying"));

    assertThat(FileSystemUtils.readContent(outLocal.getPath(), UTF_8)).isEqualTo("local bytes");
    assertThat(outLocal.getPath().isSymbolicLink()).isFalse();
    ArgumentCaptor<TreeArtifactValue> captor = ArgumentCaptor.forClass(TreeArtifactValue.class);
    verify(outputStore).injectTree(eq(out), captor.capture());
    TreeFileArtifact outRem = TreeFileArtifact.createTreeOutput(out, "rem");
    FileArtifactValue injectedLocal = captor.getValue().getChildValues().get(outLocal);
    assertThat(injectedLocal.isCopyOutput()).isTrue();
    assertThat(injectedLocal.getDigest()).isEqualTo(storeRead.getDigest());
    FileArtifactValue injectedRem = captor.getValue().getChildValues().get(outRem);
    assertThat(injectedRem.isRemote()).isTrue();
    assertThat(injectedRem.isCopyOutput()).isTrue();
    assertThat(injectedRem).isNotSameInstanceAs(remValue);
    assertThat(outRem.getPath().exists()).isFalse();
  }

  @Test
  public void localTree_copiesOnlyRecordedChildrenAndDoesNotInject() throws Exception {
    SpecialArtifact in = tree("in");
    SpecialArtifact out = tree("out");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(in, "a/f");
    TreeFileArtifact empty = TreeFileArtifact.createTreeOutput(in, "empty");
    in.getPath().getRelative("a").createDirectoryAndParents();
    FileSystemUtils.writeContent(child.getPath(), UTF_8, "x");
    FileSystemUtils.createEmptyFile(empty.getPath());
    // Not recorded: must not be copied.
    in.getPath().getRelative("stray_dir").createDirectory();
    in.getPath()
        .getRelative("stray_link")
        .createSymbolicLink(PathFragment.create("a/f"));
    when(inputProvider.getTreeMetadata(in))
        .thenReturn(
            TreeArtifactValue.newBuilder(in)
                .putChild(child, FileArtifactValue.createForNormalFile(DIGEST, null, 1))
                .putChild(empty, FileArtifactValue.createForNormalFile(DIGEST, null, 0))
                .build());

    execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying"));

    assertThat(FileSystemUtils.readContent(out.getPath().getRelative("a/f"), UTF_8)).isEqualTo("x");
    assertThat(out.getPath().getRelative("empty").isFile()).isTrue();
    assertThat(out.getPath().getRelative("stray_dir").exists()).isFalse();
    assertThat(out.getPath().getRelative("stray_link").exists()).isFalse();
    assertThat(out.getPath().getRelative("stray_link").isSymbolicLink()).isFalse();
    verify(outputStore, never()).injectTree(any(), any());
  }

  @Test
  public void emptyTree_createsEmptyOutputDirectory() throws Exception {
    SpecialArtifact in = tree("in");
    SpecialArtifact out = tree("out");
    in.getPath().createDirectoryAndParents();
    when(inputProvider.getTreeMetadata(in)).thenReturn(TreeArtifactValue.empty());

    execute(CopyAction.create(NULL_ACTION_OWNER, in, out, "Copying"));

    assertThat(out.getPath().isDirectory()).isTrue();
    assertThat(out.getPath().getDirectoryEntries()).isEmpty();
  }
}
