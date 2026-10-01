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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.OutputMetadataStore;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.WalkableGraph;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteOutputService}. */
@RunWith(JUnit4.class)
public final class RemoteOutputServiceTest {
  private final InMemoryFileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private final RemoteActionInputFetcher actionInputFetcher = mock(RemoteActionInputFetcher.class);

  private RemoteOutputService outputService;
  private ArtifactRoot outputRoot;
  private Artifact input;
  private Artifact output;
  private Artifact otherOutput;

  @Before
  public void setUp() throws Exception {
    Path execRoot = fs.getPath("/exec");
    execRoot.createDirectoryAndParents();
    outputService =
        new RemoteOutputService(
            new BlazeDirectories(
                new ServerDirectories(execRoot, execRoot, execRoot), execRoot, "bazel"),
            /* rewindLostInputs= */ true);
    outputService.setActionInputFetcher(actionInputFetcher, mock(WalkableGraph.class));
    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    input = ActionsTestUtil.createArtifact(outputRoot, "input");
    output = ActionsTestUtil.createArtifact(outputRoot, "link");
    otherOutput = ActionsTestUtil.createArtifact(outputRoot, "other_link");
  }

  @Test
  public void finalizeAction_symlinkTargetLost_reportsInputLost() throws Exception {
    SymlinkAction action =
        SymlinkAction.toArtifact(ActionsTestUtil.NULL_ACTION_OWNER, input, output, "link");
    Digest digest = digestUtil.compute("lost".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    // The failure is attributed to the input regardless of the path it names.
    doThrow(new BulkTransferException(new CacheNotFoundException(digest, otherOutput.getExecPath())))
        .when(actionInputFetcher)
        .finalizeAction(any(), any());

    var e =
        assertThrows(
            LostInputsActionExecutionException.class,
            () -> outputService.finalizeAction(action, mock(OutputMetadataStore.class)));

    assertThat(e.getLostInputs()).containsExactly(DigestUtil.toString(digest), input);
  }

  @Test
  public void finalizeAction_otherActionOutputLost_notReportedAsLostInput() throws Exception {
    Action action =
        new ActionsTestUtil.NullAction(ActionsTestUtil.NULL_ACTION_OWNER, input, output);
    Digest digest = digestUtil.compute("lost".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    BulkTransferException failure =
        new BulkTransferException(new CacheNotFoundException(digest, output.getExecPath()));
    doThrow(failure).when(actionInputFetcher).finalizeAction(any(), any());

    var e =
        assertThrows(
            BulkTransferException.class,
            () -> outputService.finalizeAction(action, mock(OutputMetadataStore.class)));

    assertThat(e).isSameInstanceAs(failure);
  }
}
