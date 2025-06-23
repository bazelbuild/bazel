// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.MainRepoMappingComputationStartingEvent;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.runtime.UiOptions.UseCurses;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UiEventHandler}. */
@RunWith(Enclosed.class)
public sealed class UiEventHandlerTest {

  private static final BuildCompleteEvent BUILD_COMPLETE_EVENT =
      new BuildCompleteEvent(new BuildResult(/* startTimeMillis= */ 0));
  private static final String BUILD_DID_NOT_COMPLETE_MESSAGE =
      "\033[31m\033[1mERROR: \033[0mBuild did NOT complete successfully" + System.lineSeparator();

  /** The escape sequence that clears the progress bar when curses is enabled. */
  private static final String CLEAR_PROGRESS_BAR = "\033[1A\033[K";

  private static final ArtifactRoot OUTPUT_ROOT =
      ArtifactRoot.asDerivedRoot(
          new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/base/exec"),
          RootType.OUTPUT,
          "out");

  @TestParameter private boolean skymeldMode;

  final UiOptions uiOptions = new UiOptions();
  final FlushCollectingOutputStream output = new FlushCollectingOutputStream();
  final ManualClock clock = new ManualClock();

  UiEventHandler uiEventHandler;

  void createUiEventHandler(EventKind outputKind) {
    uiOptions.eventKindFilters = ImmutableList.of();
    output.flush();
    output.flushed.clear();

    OutErr outErr =
        switch (outputKind) {
          case STDOUT -> OutErr.create(/* out= */ output, /* err= */ mock(OutputStream.class));
          case STDERR -> OutErr.create(/* out= */ mock(OutputStream.class), /* err= */ output);
          default -> throw new AssertionError(outputKind);
        };

    uiEventHandler =
        new UiEventHandler(
            outErr,
            uiOptions,
            /* quiet= */ false,
            clock,
            new EventBus(),
            /* workspacePathFragment= */ null,
            skymeldMode,
            /* newStatsSummary= */ false);
    uiEventHandler.mainRepoMappingComputationStarted(new MainRepoMappingComputationStartingEvent());
    uiEventHandler.buildStarted(
        BuildStartingEvent.create(
            "outputFileSystemType",
            /* usesInMemoryFileSystem= */ false,
            mock(BuildRequest.class),
            /* workspace= */ null,
            "/pwd"));
  }

  /** Test cases that exercise both stdout and stderr. */
  @RunWith(TestParameterInjector.class)
  public static final class StdoutAndStderrTest extends UiEventHandlerTest {

    @TestParameter({"STDOUT", "STDERR"})
    private EventKind outputKind;

    @Before
    public void createUiEventHandler() {
      createUiEventHandler(outputKind);
    }

    @Test
    public void buildComplete_outputsBuildFailedOnStderr() {
      uiEventHandler.buildComplete(BUILD_COMPLETE_EVENT);

      if (outputKind == EventKind.STDOUT) {
        output.assertFlushed();
      } else {
        output.assertFlushed(BUILD_DID_NOT_COMPLETE_MESSAGE);
      }
    }

    @Test
    public void buildComplete_flushesBufferedMessage() {
      uiEventHandler.handle(output("hello"));
      uiEventHandler.buildComplete(BUILD_COMPLETE_EVENT);

      if (outputKind == EventKind.STDOUT) {
        output.assertFlushed("hello");
      } else {
        output.assertFlushed("hello", System.lineSeparator() + BUILD_DID_NOT_COMPLETE_MESSAGE);
      }
    }

    @Test
    public void buildComplete_successfulBuild() {
      uiEventHandler.handle(output(""));
      var buildSuccessResult = new BuildResult(/* startTimeMillis= */ 0);
      buildSuccessResult.setDetailedExitCode(DetailedExitCode.success());
      uiEventHandler.buildComplete(new BuildCompleteEvent(buildSuccessResult));

      if (outputKind == EventKind.STDOUT) {
        output.assertFlushed();
      } else {
        output.assertFlushed(
            "\033[32mINFO: \033[0mBuild completed successfully, 0 total actions"
                + System.lineSeparator());
      }
    }

    @Test
    public void buildComplete_emptyBuffer_outputsBuildFailedOnStderr() {
      uiEventHandler.handle(output(""));
      uiEventHandler.buildComplete(BUILD_COMPLETE_EVENT);

      if (outputKind == EventKind.STDOUT) {
        output.assertFlushed();
      } else {
        output.assertFlushed(BUILD_DID_NOT_COMPLETE_MESSAGE);
      }
    }

    @Test
    public void handleOutputEvent_buffersWithoutNewline() {
      uiEventHandler.handle(output("hello"));
      output.assertFlushed();
    }

    @Test
    public void handleOutputEvent_concatenatesInBuffer() {
      uiEventHandler.handle(output("hello "));
      uiEventHandler.handle(output("there"));
      uiEventHandler.buildComplete(BUILD_COMPLETE_EVENT);

      if (outputKind == EventKind.STDOUT) {
        output.assertFlushed("hello there");
      } else {
        output.assertFlushed(
            "hello there", System.lineSeparator() + BUILD_DID_NOT_COMPLETE_MESSAGE);
      }
    }

    @Test
    public void handleOutputEvent_flushesOnNewline() {
      uiEventHandler.handle(output("hello\n"));
      output.assertFlushed("hello\n");
    }

    @Test
    public void handleOutputEvent_flushesOnlyUntilNewline() {
      uiEventHandler.handle(output("hello\nworld"));
      output.assertFlushed("hello\n");
    }

    @Test
    public void handleOutputEvent_flushesUntilLastNewline() {
      uiEventHandler.handle(output("hello\nto\neveryone"));
      output.assertFlushed("hello\nto\n");
    }

    @Test
    public void handleOutputEvent_flushesMultiLineMessageAtOnce() {
      uiEventHandler.handle(output("hello\neveryone\n"));
      output.assertFlushed("hello\neveryone\n");
    }

    @Test
    public void handleOutputEvent_concatenatesBufferBeforeFlushingOnNewline() {
      uiEventHandler.handle(output("hello"));
      uiEventHandler.handle(output(" there!\nmore text"));

      output.assertFlushed("hello there!\n");
    }

    // This test only exercises progress bar code when testing stderr output, since we don't make
    // any assertions on stderr (where the progress bar is written) when testing stdout.
    @Test
    public void noChangeOnUnflushedWrite() {
      uiOptions.showProgress = true;
      uiOptions.useCursesEnum = UseCurses.YES;
      createUiEventHandler();
      if (outputKind == EventKind.STDERR) {
        assertThat(output.flushed).hasSize(2);
        output.flushed.clear();
      }
      // Unterminated strings are saved in memory and not pushed out at all.
      assertThat(output.flushed).isEmpty();
      assertThat(output.writtenSinceFlush).isEmpty();
    }

    private Event output(String message) {
      return Event.of(outputKind, message);
    }
  }

  /** Test cases that only exercise stdout. */
  @RunWith(JUnit4.class)
  public static final class StdoutOnlyTest extends UiEventHandlerTest {

    @Before
    public void createUiEventHandler() {
      createUiEventHandler(EventKind.STDOUT);
    }

    @Test
    public void handleOutputEvent_flushesRemainingLines() {
      uiEventHandler.handle(Event.of(EventKind.STDOUT, "hello\nto\neveryone"));
      output.assertFlushed("hello\nto\n");
      uiEventHandler.afterCommand(new AfterCommandEvent());
      output.assertFlushed("hello\nto\n", "everyone");
    }
  }

  /** Test cases that only exercise stderr. */
  @RunWith(JUnit4.class)
  public static final class StderrOnlyTest extends UiEventHandlerTest {

    @Before
    public void createUiEventHandler() {
      createUiEventHandler(EventKind.STDERR);
    }

    @Test
    public void buildCompleteMessageDoesntOverrideError() {
      uiOptions.showProgress = true;
      uiOptions.useCursesEnum = UseCurses.YES;
      createUiEventHandler();

      uiEventHandler.buildComplete(BUILD_COMPLETE_EVENT);
      uiEventHandler.handle(Event.error("Show me this!"));
      uiEventHandler.afterCommand(new AfterCommandEvent());

      assertThat(output.flushed).hasSize(5);
      assertThat(output.flushed.get(3)).contains("Show me this!");
      assertThat(output.flushed.get(4)).doesNotContain(CLEAR_PROGRESS_BAR);
    }

    @Test
    public void temporarilyDisableProgress() throws Exception {
      uiOptions.showProgress = true;
      uiOptions.useCursesEnum = UseCurses.YES;
      uiOptions.showProgressRateLimit = 1;
      uiOptions.uiActionsShown = 2;
      createUiEventHandler();
      NullAction action1 = actionWithProgressMessage("Executing action 1", "action1.out");
      NullAction action2 = actionWithProgressMessage("Executing action 2", "action2.out");
      uiEventHandler.loadingComplete(
          new LoadingPhaseCompleteEvent(
              ImmutableSet.of(), ImmutableSet.of(), RepositoryMapping.EMPTY));
      uiEventHandler.analysisComplete(mock(AnalysisPhaseCompleteEvent.class));
      output.flushed.clear();

      // Showing progress, running actions shown.
      clock.advanceMillis(2000);
      uiEventHandler.runningAction(new RunningActionEvent(action1, "local"));
      assertThat(output.flushed).hasSize(1);
      assertThat(output.flushed.getFirst()).contains("Executing action 1;");

      // Disable progress, progress bar cleared.
      assertThat(uiEventHandler.disableProgress()).isTrue();
      assertThat(output.flushed).hasSize(2);
      assertThat(output.flushed.getLast()).endsWith(CLEAR_PROGRESS_BAR);

      // Another action starts running, still no progress updates.
      clock.advanceMillis(2000);
      uiEventHandler.runningAction(new RunningActionEvent(action2, "local"));
      assertThat(output.flushed).hasSize(2);

      // Enable progress again, progress bar written with both running actions.
      uiEventHandler.enableProgress();
      assertThat(output.flushed).hasSize(3);
      assertThat(output.flushed.getLast()).contains("2 actions running");
      assertThat(output.flushed.getLast()).contains("Executing action 1;");
      assertThat(output.flushed.getLast()).contains("Executing action 2;");
    }

    @Test
    public void progressOff_disableProgressReturnsFalse() throws Exception {
      uiOptions.showProgress = false;
      createUiEventHandler();
      assertThat(uiEventHandler.disableProgress()).isFalse();
    }

    @Test
    public void progressAlreadyDisabled_disableProgressReturnsFalse() throws Exception {
      uiOptions.showProgress = true;
      createUiEventHandler();
      assertThat(uiEventHandler.disableProgress()).isTrue();
      assertThat(uiEventHandler.disableProgress()).isFalse();
    }

    private static NullAction actionWithProgressMessage(String progressMessage, String outputPath) {
      Artifact output = ActionsTestUtil.createArtifact(OUTPUT_ROOT, outputPath);
      return new NullAction(output) {
        @Override
        protected String getRawProgressMessage() {
          return progressMessage;
        }
      };
    }
  }

  private static final class FlushCollectingOutputStream extends OutputStream {
    private final List<String> flushed = new ArrayList<>();
    private String writtenSinceFlush = "";

    @Override
    public void write(int b) throws IOException {
      write(new byte[] {(byte) b});
    }

    @Override
    public void write(byte[] bytes, int offset, int len) {
      writtenSinceFlush += new String(Arrays.copyOfRange(bytes, offset, offset + len), UTF_8);
    }

    @Override
    public void flush() {
      // Ignore inconsequential extra flushes.
      if (!writtenSinceFlush.isEmpty()) {
        flushed.add(writtenSinceFlush);
      }
      writtenSinceFlush = "";
    }

    private void assertFlushed(String... messages) {
      assertThat(writtenSinceFlush).isEmpty();
      assertThat(flushed).containsExactlyElementsIn(messages);
    }
  }
}
