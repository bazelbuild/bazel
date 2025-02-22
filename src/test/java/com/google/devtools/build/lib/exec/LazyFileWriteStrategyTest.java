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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.FakeInputMetadataHandlerBase;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.remote.RemoteActionFileSystem;
import com.google.devtools.build.lib.remote.RemoteActionInputFetcher;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link LazyFileWriteStrategy}. */
@RunWith(TestParameterInjector.class)
public final class LazyFileWriteStrategyTest {
  private static final IOException INJECTED_EXCEPTION = new IOException("oh no!");

  private final LazyFileWriteStrategy lazyFileWriteStrategy = new LazyFileWriteStrategy();

  private final FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Scratch scratch = new Scratch(fileSystem);
  private Path execRoot;
  private ArtifactRoot outputRoot;
  private RemoteActionFileSystem actionFileSystem;
  private FakeInputMetadataHandlerBase metadataHandler;
  private AbstractAction action;

  @Before
  public void createOutputRoot() throws IOException {
    execRoot = scratch.dir("/execroot");
    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "bazel-out");
    outputRoot.getRoot().asPath().createDirectory();
    metadataHandler =
        new FakeInputMetadataHandlerBase() {
          private Map<ActionInput, FileArtifactValue> injectedMetadata = new HashMap<>();

          @Override
          public void injectFile(Artifact output, FileArtifactValue metadata) {
            injectedMetadata.put(output, metadata);
          }

          @Nullable
          @Override
          public FileArtifactValue getOutputMetadata(ActionInput input) {
            return injectedMetadata.get(input);
          }
        };
    action = createAction("file");
    actionFileSystem =
        new RemoteActionFileSystem(
            fileSystem,
            execRoot.asFragment(),
            outputRoot.getExecPathString(),
            new ActionInputMap(BugReporter.defaultInstance(), 0),
            action.getOutputs(),
            metadataHandler,
            mock(RemoteActionInputFetcher.class));
    actionFileSystem.createDirectoryAndParents(
        execRoot
            .getRelative(action.getPrimaryOutput().getExecPath().getParentDirectory())
            .asFragment());
  }

  @Test
  public void writeOutputToFile_writesCorrectOutput(
      @TestParameter({"", "hello", "hello there"}) String content) throws Exception {
    var unused =
        lazyFileWriteStrategy.writeOutputToFile(
            action,
            createActionExecutionContext(actionFileSystem, metadataHandler),
            out -> out.write(content.getBytes(UTF_8)),
            /* makeExecutable= */ false,
            /* isRemotable= */ true);

    FileArtifactValue metadata = metadataHandler.getOutputMetadata(action.getPrimaryOutput());
    assertThat(metadata.isInline()).isTrue();
    assertThat(metadata.getInputStream().readAllBytes()).isEqualTo(content.getBytes(UTF_8));
  }

  @Test
  public void writeOutputToFile_errorInWriter_returnsFailure() throws Exception {
    AbstractAction action = createAction("file");

    ExecException e =
        assertThrows(
            EnvironmentalExecException.class,
            () ->
                lazyFileWriteStrategy.writeOutputToFile(
                    action,
                    createActionExecutionContext(actionFileSystem, metadataHandler),
                    out -> {
                      throw INJECTED_EXCEPTION;
                    },
                    /* makeExecutable= */ false,
                    /* isRemotable= */ true));

    assertThat(e).hasCauseThat().isSameInstanceAs(INJECTED_EXCEPTION);
    var detailExitCode = getDetailExitCode(e);
    assertThat(detailExitCode.getExitCode()).isEqualTo(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    assertThat(detailExitCode.getFailureDetail().getExecution().getCode())
        .isEqualTo(Code.FILE_WRITE_IO_EXCEPTION);
  }

  private DetailedExitCode getDetailExitCode(ExecException e) {
    return ActionExecutionException.fromExecException(e, new NullAction()).getDetailedExitCode();
  }

  private ActionExecutionContext createActionExecutionContext(
      FileSystem actionFileSystem, FakeInputMetadataHandlerBase metadataHandler) {
    Executor executor = new DummyExecutor(fileSystem, execRoot);
    return new ActionExecutionContext(
        executor,
        /* inputMetadataProvider= */ metadataHandler,
        ActionInputPrefetcher.NONE,
        new ActionKeyContext(),
        /* outputMetadataStore= */ metadataHandler,
        /* rewindingEnabled= */ false,
        ActionExecutionContext.LostInputsCheck.NONE,
        /* fileOutErr= */ null,
        NullEventHandler.INSTANCE,
        /* clientEnv= */ ImmutableMap.of(),
        /* topLevelFilesets= */ ImmutableMap.of(),
        treeArtifact -> ImmutableSortedSet.of(),
        /* actionFileSystem= */ actionFileSystem,
        /* skyframeDepsResult= */ null,
        DiscoveredModulesPruner.DEFAULT,
        SyscallCache.NO_CACHE,
        ThreadStateReceiver.NULL_INSTANCE);
  }

  private AbstractAction createAction(String outputRelativePath) {
    return new NullAction(
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create(outputRelativePath)));
  }
}
