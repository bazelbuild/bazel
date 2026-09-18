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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.InputFileErrorException;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.TopLevelOutputException;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.constraints.TopLevelConstraintSemantics.TargetCompatibilityCheckException;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.causes.LoadingFailedCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadingFailureEvent;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.skyframe.ArtifactNestedSetFunction.ArtifactNestedSetEvalException;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue.AspectCompletionKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.TopLevelAspectsKey;
import com.google.devtools.build.lib.skyframe.SkyframeErrorProcessor.ErrorProcessingResult;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Characterization tests for {@link SkyframeErrorProcessor}.
 *
 * <p><b>These are characterization tests.</b> They pin down what the class does <em>today</em>, not
 * what it arguably should do. Where today's behavior looks like a wart, it is still pinned, with a
 * comment naming the wart. A failure here after a refactoring means the refactoring changed
 * observable behavior - that may be intentional, but it must be a conscious decision.
 *
 * <p>Only the entry points {@link SkyframeErrorProcessor#processErrors}, {@link
 * SkyframeErrorProcessor#processAnalysisErrors} and {@link
 * SkyframeErrorProcessor#processExecutionErrors} are exercised; private helpers are deliberately
 * never referenced.
 *
 * <p><b>Order dependence:</b> {@link EvaluationResult.Builder} stores errors in a {@link
 * java.util.HashMap}, so with more than one error the iteration order of {@code
 * EvaluationResult#errorMap} is hash-dependent. Tests that would otherwise depend on that order
 * assert with {@code isAnyOf} and say so in a comment.
 */
@RunWith(TestParameterInjector.class)
public class SkyframeErrorProcessorTest {

  private static final TopLevelArtifactContext TOP_LEVEL_ARTIFACT_CONTEXT =
      new TopLevelArtifactContext(
          /* runTestsExclusively= */ false,
          /* outputGroups= */ ImmutableSortedSet.of(),
          /* failOnUnknownOutputGroups= */ false,
          /* forRunCommand= */ false);

  private static final AspectClass ASPECT_CLASS = () -> "TestAspect";

  /** {@code SkyframeErrorProcessor#CYCLE_CODE}, duplicated here on purpose: it is a contract. */
  private static final DetailedExitCode EXECUTION_CYCLE_CODE =
      DetailedExitCode.of(
          FailureDetail.newBuilder()
              .setMessage("cycle found during execution")
              .setExecution(Execution.newBuilder().setCode(Execution.Code.CYCLE))
              .build());

  private final RecordingCyclesReporter cyclesReporter = new RecordingCyclesReporter();
  private final StoredEventHandler eventHandler = new StoredEventHandler();
  private final RecordingBugReporter bugReporter = new RecordingBugReporter();
  private final EventBus eventBus = new EventBus();
  private final EventBusCollector eventBusCollector = new EventBusCollector();

  public SkyframeErrorProcessorTest() {
    eventBus.register(eventBusCollector);
  }

  // -------------------------------------------------------------------------------------------
  // A. Classification by exception type (on a ConfiguredTargetKey).
  // -------------------------------------------------------------------------------------------

  @Test
  public void configuredValueCreationException_analysisCausesOnly_keepGoing_analysisError()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//analysis_err");

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(analysisException("analysis exception", key.getLabel()))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(result.actionConflicts()).isEmpty();
    assertThat(result.aspectKeysForConflictReporting()).isEmpty();
  }

  @Test
  public void configuredValueCreationException_loadingCauses_keepGoing_loadingAndAnalysisError()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//loading_err");
    Label loadingRootCause = Label.parseCanonicalUnchecked("//missing_dep");

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    analysisExceptionWithCauses(
                        "loading exception",
                        key.getLabel(),
                        // Two distinct Causes with the *same* label: they must be de-duplicated.
                        NestedSetBuilder.create(
                            Order.STABLE_ORDER,
                            new LoadingFailedCause(loadingRootCause, analysisExitCode("first")),
                            new LoadingFailedCause(
                                loadingRootCause, analysisExitCode("second")))))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasLoadingError()).isTrue();
    // A loading error is also an analysis error: hasAnalysisError is true for everything that is
    // not an execution error.
    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
    // The de-duplication itself is observable through the LoadingFailureEvents, see
    // loadingError_postsOneLoadingFailureEventPerDedupedLabel below.
  }

  @Test
  public void noSuchTargetException_keepGoing_analysisError() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//no_such_target");

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key, errorInfo(new NoSuchTargetException(key.getLabel(), "target doesn't exist"))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(result.executionDetailedExitCode()).isNull();
  }

  @Test
  public void noSuchTargetException_keepGoing_rootCauseIsAnalysisFailedCauseFromException()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//no_such_target");
    NoSuchTargetException cause = new NoSuchTargetException(key.getLabel(), "target doesn't exist");

    processErrors(
        resultOf(key, errorInfo(cause)), /* keepGoing= */ true, /* includeExecutionPhase= */ false);

    AnalysisFailureEvent event = onlyAnalysisFailureEvent();
    assertThat(event.getFailedTarget()).isEqualTo(key);
    assertThat(rootCauseDetailedExitCodes(event)).containsExactly(cause.getDetailedExitCode());
  }

  @Test
  public void externalDepsException_keepGoing_analysisErrorWithCauseFromException()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//external_deps_err");
    ExternalDepsException cause =
        ExternalDepsException.withMessage(ExternalDeps.Code.BAD_MODULE, "bad module");

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(cause)),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(rootCauseDetailedExitCodes(onlyAnalysisFailureEvent()))
        .containsExactly(cause.getDetailedExitCode());
  }

  @Test
  public void targetCompatibilityCheckException_keepGoing_analysisErrorWithEmptyRootCauses()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//incompatible");

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    new TargetCompatibilityCheckException(
                        "incompatible target", analysisFailureDetail("incompatible target")))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(onlyAnalysisFailureEvent().getRootCauses().toList()).isEmpty();
  }

  @Test
  public void actionExecutionException_keepGoing_executionErrorWithExitCodeFromException()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(actionExecutionException("action failed", exitCode))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(exitCode);
    // An execution error is *not* an analysis error.
    assertThat(result.hasAnalysisError()).isFalse();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(bugReporter.nonFatalBugReports).isEmpty();
  }

  @Test
  public void actionExecutionException_keepGoing_rootCausesComeFromGetRootCauses()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    Label rootCauseLabel = Label.parseCanonicalUnchecked("//failing_action_owner");
    ActionExecutionException cause =
        new ActionExecutionException(
            "action failed",
            /* action= */ (ActionAnalysisMetadata) null,
            /* rootCauses= */ NestedSetBuilder.<Cause>create(
                Order.STABLE_ORDER,
                new LabelCause(rootCauseLabel, analysisExitCode("failing action"))),
            /* catastrophe= */ false,
            executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE));

    // No event bus: the root causes are then emitted to the event handler, which is the only way
    // to observe them for an execution error (no AnalysisFailureEvent is posted for those).
    processErrorsInBuildViewTest(
        resultOf(key, errorInfo(cause)), /* keepGoing= */ true, /* includeExecutionPhase= */ true);

    assertThat(errorMessages()).hasSize(1);
    assertThat(errorMessages().get(0)).contains(rootCauseLabel.toString());
  }

  @Test
  public void inputFileErrorException_keepGoing_executionError() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//input_file_err");
    DetailedExitCode exitCode =
        executionExitCode("missing input", Execution.Code.SOURCE_INPUT_MISSING);

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(new InputFileErrorException("missing input", exitCode))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(exitCode);
    assertThat(result.hasAnalysisError()).isFalse();
  }

  @Test
  public void topLevelOutputException_keepGoing_executionError() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//top_level_output_err");
    DetailedExitCode exitCode =
        executionExitCode("bad top-level output", Execution.Code.ACTION_OUTPUTS_NOT_CREATED);

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(new TopLevelOutputException("bad top-level output", exitCode))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(exitCode);
    assertThat(result.hasAnalysisError()).isFalse();
  }

  @Test
  public void testExecException_keepGoing_executionErrorWithUnknownExitCodeAndBugReport()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//test_err");
    // TestExecException is an execution exception but is *not* a DetailedException, so the
    // processor files a non-fatal bug report and synthesizes an UNEXPECTED_EXCEPTION exit code.
    TestExecException cause =
        new TestExecException("test failed", FailureDetail.getDefaultInstance());

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(cause)),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.hasAnalysisError()).isFalse();
    assertThat(result.executionDetailedExitCode())
        .isEqualTo(
            executionExitCode(
                "Unexpected exception, please file an issue with the Bazel team: test failed",
                Execution.Code.UNEXPECTED_EXCEPTION));
    assertThat(bugReporter.nonFatalBugReports).hasSize(1);
  }

  @Test
  public void artifactNestedSetEvalException_keepGoing_unknownExitCodeAndBugReport()
      throws Exception {
    // The fifth and last isExecutionException type, and the other one that is not a
    // DetailedException: same bug report, same synthesized UNEXPECTED_EXCEPTION exit code.
    ConfiguredTargetKey key = configuredTargetKey("//nested_set_err");

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    artifactNestedSetEvalException(
                        "nested set failed", /* catastrophic= */ false))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.hasAnalysisError()).isFalse();
    assertThat(result.executionDetailedExitCode())
        .isEqualTo(
            executionExitCode(
                "Unexpected exception, please file an issue with the Bazel team: nested set failed",
                Execution.Code.UNEXPECTED_EXCEPTION));
    assertThat(bugReporter.nonFatalBugReports).hasSize(1);
  }

  @Test
  public void unrecognizedExceptionType_crashesWithABugReport() {
    // In production, BugReport.logUnexpected just logs, and the error would end up classified as
    // an analysis error with empty root causes. In a test, BugReport turns that into a thrown
    // IllegalStateException, which happens before any classification. Pinned as-is: the point is
    // that this code path files a bug report.
    ConfiguredTargetKey key = configuredTargetKey("//unrecognized");

    EvaluationResult<SkyValue> result =
        resultOf(key, errorInfo(new UnrecognizedException("not a known type")));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> processErrors(result, /* keepGoing= */ true, /* includeExecutionPhase= */ false));

    // keepEdges is false in the helper, so the processor takes the "direct deps not stored" path
    // instead of walking a (missing) WalkableGraph.
    assertThat(thrown).hasMessageThat().contains("Unexpected analysis error");
    assertThat(thrown).hasMessageThat().contains("direct deps not stored");
  }

  // -------------------------------------------------------------------------------------------
  // B. Cycles.
  // -------------------------------------------------------------------------------------------

  @Test
  public void analysisCycle_keepGoing_analysisErrorWithCycleLabelCause() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//pkg:cycle");
    ConfiguredTargetKey culprit = configuredTargetKey("//cycle:culprit");

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key, ErrorInfo.fromCycle(CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();

    // The warning only appends "\n" + cause.getMessage() when the cause is non-null, and
    // ErrorInfo.fromCycle forces a null exception, so a pure cycle warning never has a trailing
    // cause suffix. Pinned with exact equality: that is the only way to pin an *absent* suffix.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:cycle', it will not be built.");

    AnalysisFailureEvent event = onlyAnalysisFailureEvent();
    assertThat(event.getRootCauses().toList())
        .containsExactly(
            new LabelCause(
                culprit.getLabel(),
                DetailedExitCode.of(
                    FailureDetail.newBuilder()
                        .setMessage("Dependency cycle")
                        .setAnalysis(Analysis.newBuilder().setCode(Analysis.Code.CYCLE))
                        .build())));
  }

  @Test
  public void analysisCycle_noKeepGoing_throwsViewCreationFailedException() {
    ConfiguredTargetKey key = configuredTargetKey("//pkg:cycle");
    ConfiguredTargetKey culprit = configuredTargetKey("//cycle:culprit");

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        ErrorInfo.fromCycle(CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ false));

    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo("Analysis of target '//pkg:cycle' failed; build aborted");
    assertThat(thrown.getFailureDetail().getAnalysis().getCode()).isEqualTo(Analysis.Code.CYCLE);
    // The failure detail's message carries a " due to cycle" suffix that the exception's own
    // message does not: createViewCreationFailedException uses the 2-arg constructor for a cycle,
    // which does not touch the message. Both are asserted to keep the contrast visible.
    assertThat(thrown.getFailureDetail().getMessage())
        .isEqualTo("Analysis of target '//pkg:cycle' failed; build aborted due to cycle");
  }

  // TODO(b/561978611): Remove this behavior. An execution cycle is reported with analysis wording,
  // because its cause is null.
  @Test
  public void executionCycle_keepGoing_executionErrorWithCycleCode() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//pkg:exec_cycle");

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, ErrorInfo.fromCycle(executionCycle(key))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(EXECUTION_CYCLE_CODE);
    assertThat(result.hasAnalysisError()).isFalse();
    // WART, pinned deliberately: the wording is wrong for an execution cycle. processErrors
    // computes isExecutionException from the *cause*, and a cycle has a null cause, so it is false
    // here even though the exit code is the execution CYCLE code and isAnalysisError() is false.
    // logOrPrintWarningsKeepGoing therefore emits the analysis-worded warning.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:exec_cycle', it will not be built.");
  }

  @Test
  public void executionCycle_noKeepGoing_throwsBuildFailedExceptionWithCycleCode() {
    ConfiguredTargetKey key = configuredTargetKey("//exec_cycle");

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(key, ErrorInfo.fromCycle(executionCycle(key))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    // StarlarkExecutionTests asserts on exactly this code: do not change it lightly.
    assertThat(thrown.getDetailedExitCode()).isEqualTo(EXECUTION_CYCLE_CODE);
    assertThat(thrown.getDetailedExitCode().getFailureDetail().getExecution().getCode())
        .isEqualTo(Execution.Code.CYCLE);
    assertThat(thrown).hasMessageThat().isNull();
  }

  @Test
  public void cycle_reportsCyclesWithTheOriginalPreUnwrapKey() throws Exception {
    ConfiguredTargetKey ctKey = configuredTargetKey("//cycle");
    SkyKey originalKey =
        BuildDriverKey.ofConfiguredTarget(
            ctKey,
            TOP_LEVEL_ARTIFACT_CONTEXT,
            /* explicitlyRequested= */ true,
            /* skipIncompatibleExplicitTargets= */ false,
            /* extraActionTopLevelOnly= */ false,
            /* keepGoing= */ true);
    CycleInfo cycleInfo =
        CycleInfo.createCycleInfo(ImmutableList.of(configuredTargetKey("//cycle:culprit")));

    processErrors(
        resultOf(originalKey, ErrorInfo.fromCycle(cycleInfo)),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ true);

    assertThat(cyclesReporter.topLevelKeys).containsExactly(originalKey);
    assertThat(cyclesReporter.cycles).containsExactly(cycleInfo);
  }

  // -------------------------------------------------------------------------------------------
  // C. Key normalization.
  // -------------------------------------------------------------------------------------------

  /** The wrapper key types that {@code processErrors} peels away before classifying. */
  private enum TopLevelKeyKind {
    BARE_CONFIGURED_TARGET,
    BUILD_DRIVER,
    TARGET_COMPLETION,
    TEST_COMPLETION,
    ASPECT_COMPLETION
  }

  @Test
  public void keyNormalization_sameErrorProcessingResultForEveryWrapperKind(
      @TestParameter TopLevelKeyKind kind) throws Exception {
    ConfiguredTargetKey ctKey = configuredTargetKey("//wrapped");
    SkyKey key = wrapKey(kind, ctKey);

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(analysisException("analysis exception", ctKey.getLabel()))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    // Note: the *events* differ between these (an AspectCompletionKey unwraps to a bare AspectKey,
    // which gets no event and discards its root causes) - see section D. The ErrorProcessingResult
    // is identical.
    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(result.actionConflicts()).isEmpty();
    assertThat(result.aspectKeysForConflictReporting()).isEmpty();
  }

  // TODO(b/561978611): Remove this behavior. The warning names the target 'null', because an
  // ActionLookupData has no label.
  @Test
  public void actionLookupDataKey_isAlwaysAnExecutionError() throws Exception {
    ConfiguredTargetKey ctKey = configuredTargetKey("//build_info");
    ActionLookupData key = ActionLookupData.create(ctKey, /* actionIndex= */ 0);
    // Deliberately an *analysis* exception: for an ActionLookupData key it is still classified as
    // an execution error, and its DetailedExitCode is used as the execution exit code.
    ConfiguredValueCreationException cause =
        analysisException("analysis exception", ctKey.getLabel());

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(cause)),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(cause.getDetailedExitCode());
    assertThat(result.hasAnalysisError()).isFalse();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(eventBusCollector.allEvents).isEmpty();
    // WART: ActionLookupData implements ExecutionPhaseSkyKey, not ActionLookupKey, so the label
    // lookup returns null and the warning interpolates the literal string "null". The analysis
    // wording is itself a second wart here (see executionCycle_keepGoing_...): the cause is an
    // analysis exception, so isExecutionException is false even though this key is always
    // classified as an execution error.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target 'null', it will not be built.\n"
                + "analysis exception");
  }

  // -------------------------------------------------------------------------------------------
  // D. Event posting contract (real EventBus).
  // -------------------------------------------------------------------------------------------

  @Test
  public void analysisError_postsExactlyOneAnalysisFailureEventAttributedToTheKey()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//analysis_err");
    Label rootCauseLabel = Label.parseCanonicalUnchecked("//analysis_err:dep");
    LabelCause rootCause = new LabelCause(rootCauseLabel, analysisExitCode("dep failed"));

    processErrors(
        resultOf(
            key,
            errorInfo(
                analysisExceptionWithCauses(
                    "analysis exception",
                    key.getLabel(),
                    NestedSetBuilder.create(Order.STABLE_ORDER, rootCause)))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ false);

    AnalysisFailureEvent event = onlyAnalysisFailureEvent();
    assertThat(event.getFailedTarget()).isEqualTo(key);
    assertThat(event.getRootCauses().toList()).containsExactly(rootCause);
    assertThat(eventBusCollector.loadingFailures).isEmpty();
  }

  @Test
  public void loadingError_postsOneLoadingFailureEventPerDedupedLabel() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//loading_err");
    Label firstRootCause = Label.parseCanonicalUnchecked("//missing_a");
    Label secondRootCause = Label.parseCanonicalUnchecked("//missing_b");

    processErrors(
        resultOf(
            key,
            errorInfo(
                analysisExceptionWithCauses(
                    "loading exception",
                    key.getLabel(),
                    NestedSetBuilder.create(
                        Order.STABLE_ORDER,
                        // Same label twice with different exit codes: one event, not two.
                        new LoadingFailedCause(firstRootCause, analysisExitCode("first")),
                        new LoadingFailedCause(firstRootCause, analysisExitCode("second")),
                        new LoadingFailedCause(secondRootCause, analysisExitCode("third")))))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ false);

    // The loading root causes are collected in a HashSet, so the order of the events is not
    // deterministic; containsExactly is order-independent.
    assertThat(eventBusCollector.loadingFailures)
        .containsExactly(
            new LoadingFailureEvent(key.getLabel(), firstRootCause),
            new LoadingFailureEvent(key.getLabel(), secondRootCause));
    // Plus the AnalysisFailureEvent.
    assertThat(eventBusCollector.analysisFailures).hasSize(1);
  }

  @Test
  public void topLevelAspectsKey_postsAnalysisFailureEventForBaseTargetWithEmptyRootCauses()
      throws Exception {
    Label label = Label.parseCanonicalUnchecked("//aspect_err");
    TopLevelAspectsKey key = topLevelAspectsKey(label);

    processErrors(
        resultOf(
            key,
            errorInfo(
                analysisExceptionWithCauses(
                    "aspect analysis exception",
                    label,
                    NestedSetBuilder.create(
                        Order.STABLE_ORDER,
                        new LabelCause(label, analysisExitCode("aspect failed")))))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ false);

    AnalysisFailureEvent event = onlyAnalysisFailureEvent();
    assertThat(event.getFailedTarget()).isEqualTo(key.getBaseConfiguredTargetKey());
    // TODO(b/561978611): aspect errors discard their root causes today. This assertion is exactly
    // what will catch a change to that.
    assertThat(event.getRootCauses().toList()).isEmpty();
  }

  @Test
  public void bareAspectKey_postsNoEventAtAll() throws Exception {
    ConfiguredTargetKey baseKey = configuredTargetKey("//aspect_err");
    AspectKey key = aspectKey(baseKey);

    processErrors(
        resultOf(
            key, errorInfo(analysisException("aspect analysis exception", baseKey.getLabel()))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ false);

    assertThat(eventBusCollector.allEvents).isEmpty();
  }

  @Test
  public void executionError_postsNoAnalysisFailureEvent() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");

    processErrors(
        resultOf(
            key,
            errorInfo(
                actionExecutionException(
                    "action failed",
                    executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE)))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ true);

    assertThat(eventBusCollector.analysisFailures).isEmpty();
    assertThat(eventBusCollector.allEvents).isEmpty();
  }

  @Test
  public void actionConflict_postsNoEventAtAll(@TestParameter boolean keepGoing) throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//conflict");

    // Deferred to SkyframeBuildView, which has the downstream information needed to report it.
    processErrors(
        resultOf(key, errorInfo(actionConflictException("conflict"))),
        keepGoing,
        /* includeExecutionPhase= */ false);

    assertThat(eventBusCollector.allEvents).isEmpty();
    assertThat(eventHandler.getEvents()).isEmpty();
  }

  @Test
  public void nullEventBus_emitsRootCausesToTheEventHandlerAndDoesNotCrash() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//analysis_err");
    Label rootCauseLabel = Label.parseCanonicalUnchecked("//analysis_err:dep");

    processErrorsInBuildViewTest(
        resultOf(
            key,
            errorInfo(
                analysisExceptionWithCauses(
                    "analysis exception",
                    key.getLabel(),
                    NestedSetBuilder.create(
                        Order.STABLE_ORDER,
                        new LabelCause(rootCauseLabel, analysisExitCode("dep failed")))))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ false);

    assertThat(errorMessages()).hasSize(1);
    assertThat(errorMessages().get(0)).contains(rootCauseLabel.toString());
  }

  // -------------------------------------------------------------------------------------------
  // E. --nokeep_going throwing and precedence.
  // -------------------------------------------------------------------------------------------

  @Test
  public void noKeepGoing_singleAnalysisError_throwsViewCreationFailedWithPinnedMessage(
      @TestParameter boolean includeExecutionPhase) throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//pkg:analysis_err");
    ConfiguredValueCreationException cause =
        analysisException("analysis exception", key.getLabel());

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                processErrors(
                    resultOf(key, errorInfo(cause)),
                    /* keepGoing= */ false,
                    includeExecutionPhase));

    // BuildViewTest has five assertions on this exact string.
    assertThat(thrown)
        .hasMessageThat()
        .contains("Analysis of target '//pkg:analysis_err' failed; build aborted");
    assertThat(thrown).hasCauseThat().isEqualTo(cause);
  }

  @Test
  public void noKeepGoing_singleAspectAnalysisError_throwsViewCreationFailedWithAspectMessage()
      throws Exception {
    Label label = Label.parseCanonicalUnchecked("//aspect_err");
    TopLevelAspectsKey key = topLevelAspectsKey(label);
    ConfiguredValueCreationException cause = analysisException("aspect analysis exception", label);

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                processErrors(
                    resultOf(key, errorInfo(cause)),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ false));

    assertThat(thrown)
        .hasMessageThat()
        .contains(
            String.format("Analysis of aspects '%s' failed; build aborted", key.getDescription()));
    assertThat(thrown).hasCauseThat().isEqualTo(cause);
  }

  // TODO(b/561978611): Remove this behavior. Failing a build with a raw ClassCastException tells
  // the user nothing.
  @Test
  public void noKeepGoing_bareAspectKeyAnalysisError_throwsClassCastException() {
    // Known wart: throwOrReturnAspectAnalysisException only special-cases TopLevelAspectsKey, so a
    // bare AspectKey falls through to an unchecked cast to ConfiguredTargetKey.
    ConfiguredTargetKey baseKey = configuredTargetKey("//aspect_err");
    AspectKey key = aspectKey(baseKey);

    EvaluationResult<SkyValue> result =
        resultOf(key, errorInfo(analysisException("aspect exception", baseKey.getLabel())));

    assertThrows(
        ClassCastException.class,
        () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ false));
  }

  @Test
  public void noKeepGoing_singleActionExecutionError_throwsBuildFailedException() {
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        errorInfo(actionExecutionExceptionWithAction("action failed", exitCode))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown.getDetailedExitCode()).isEqualTo(exitCode);
    // rethrow() prefixes the message with the action's description. The action's owner has no
    // location, so no location prefix is added.
    assertThat(thrown).hasMessageThat().isEqualTo("TestAction failed: action failed");
  }

  @Test
  public void noKeepGoing_inputFileErrorException_isRethrownAsIs() {
    ConfiguredTargetKey key = configuredTargetKey("//input_file_err");
    InputFileErrorException cause =
        new InputFileErrorException(
            "missing input",
            executionExitCode("missing input", Execution.Code.SOURCE_INPUT_MISSING));

    InputFileErrorException thrown =
        assertThrows(
            InputFileErrorException.class,
            () ->
                processErrors(
                    resultOf(key, errorInfo(cause)),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown).isSameInstanceAs(cause);
  }

  @Test
  public void noKeepGoing_topLevelOutputException_isRethrownAsIs() {
    ConfiguredTargetKey key = configuredTargetKey("//top_level_output_err");
    TopLevelOutputException cause =
        new TopLevelOutputException(
            "bad output",
            executionExitCode("bad output", Execution.Code.ACTION_OUTPUTS_NOT_CREATED));

    TopLevelOutputException thrown =
        assertThrows(
            TopLevelOutputException.class,
            () ->
                processErrors(
                    resultOf(key, errorInfo(cause)),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown).isSameInstanceAs(cause);
  }

  @Test
  public void noKeepGoing_testExecExceptionNestedInActionExecutionException_isRethrown() {
    ConfiguredTargetKey key = configuredTargetKey("//test_err");
    TestExecException testExecException =
        new TestExecException("test failed", FailureDetail.getDefaultInstance());
    ActionExecutionException cause =
        new ActionExecutionException(
            "action failed",
            testExecException,
            /* action= */ (ActionAnalysisMetadata) null,
            /* catastrophe= */ false,
            executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE));

    TestExecException thrown =
        assertThrows(
            TestExecException.class,
            () ->
                processErrors(
                    resultOf(key, errorInfo(cause)),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown).isSameInstanceAs(testExecException);
  }

  @Test
  public void noKeepGoing_bareTestExecException_becomesBuildFailedExceptionWithBugReport() {
    // Note: a *bare* TestExecException is not rethrown; only a TestExecException that is the cause
    // of another execution exception is (see above). It is also not a DetailedException, so a bug
    // report is filed and an UNEXPECTED_EXCEPTION exit code is synthesized.
    ConfiguredTargetKey key = configuredTargetKey("//test_err");
    TestExecException cause =
        new TestExecException("test failed", FailureDetail.getDefaultInstance());

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(key, errorInfo(cause)),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown.getDetailedExitCode().getFailureDetail().getExecution().getCode())
        .isEqualTo(Execution.Code.UNEXPECTED_EXCEPTION);
    // The bug report is filed twice: once while classifying the error and once again in rethrow().
    assertThat(bugReporter.nonFatalBugReports).hasSize(2);
  }

  @Test
  public void noKeepGoing_aspectErrorPlusTargetAnalysisError_targetErrorWins() {
    // Deterministic despite the HashMap iteration order: the aspect's exception is stashed and only
    // thrown after the loop, so the target's exception always wins.
    Label aspectLabel = Label.parseCanonicalUnchecked("//aspect_err");
    TopLevelAspectsKey aspectKey = topLevelAspectsKey(aspectLabel);
    ConfiguredTargetKey targetKey = configuredTargetKey("//pkg:analysis_err");

    EvaluationResult<SkyValue> result =
        EvaluationResult.<SkyValue>builder()
            .addError(aspectKey, errorInfo(analysisException("aspect exception", aspectLabel)))
            .addError(
                targetKey, errorInfo(analysisException("target exception", targetKey.getLabel())))
            .build();

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ false));

    assertThat(thrown)
        .hasMessageThat()
        .contains("Analysis of target '//pkg:analysis_err' failed; build aborted");
  }

  @Test
  public void noKeepGoing_aspectErrorPlusActionConflict_throwsAspectErrorAndDropsTheConflict() {
    // Known wart: the action conflict is aggregated into a result that is then never returned,
    // because the stashed aspect exception is thrown after the loop. The conflict is silently lost.
    Label aspectLabel = Label.parseCanonicalUnchecked("//aspect_err");
    TopLevelAspectsKey aspectKey = topLevelAspectsKey(aspectLabel);
    ConfiguredTargetKey conflictKey = configuredTargetKey("//conflict");

    EvaluationResult<SkyValue> result =
        EvaluationResult.<SkyValue>builder()
            .addError(aspectKey, errorInfo(analysisException("aspect exception", aspectLabel)))
            .addError(conflictKey, errorInfo(actionConflictException("conflict")))
            .build();

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ false));

    assertThat(thrown)
        .hasMessageThat()
        .contains(
            String.format(
                "Analysis of aspects '%s' failed; build aborted", aspectKey.getDescription()));
  }

  @Test
  public void noKeepGoing_executionErrorPlusAnalysisError_winnerIsOrderDependent() {
    // ORDER DEPENDENCE: both errors throw immediately when they are reached, so whichever one
    // errorMap() happens to yield first wins. Do not pin a winner here.
    ConfiguredTargetKey executionKey = configuredTargetKey("//exec_err");
    ConfiguredTargetKey analysisKey = configuredTargetKey("//analysis_err");

    EvaluationResult<SkyValue> result =
        EvaluationResult.<SkyValue>builder()
            .addError(
                executionKey,
                errorInfo(
                    actionExecutionExceptionWithAction(
                        "action failed",
                        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE))))
            .addError(
                analysisKey,
                errorInfo(analysisException("analysis exception", analysisKey.getLabel())))
            .build();

    Exception thrown =
        assertThrows(
            Exception.class,
            () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ true));

    assertThat(thrown.getClass())
        .isAnyOf(BuildFailedException.class, ViewCreationFailedException.class);
  }

  @Test
  public void noKeepGoing_executionErrorPlusAspectAnalysisError_executionErrorWins() {
    // The mirror image of the test above, and this pairing *is* deterministic: an aspect's
    // exception is only stashed, so the loop carries on and reaches the execution error, which
    // rethrow() throws on the spot. The stashed exception's post-loop throw is never reached,
    // whichever order errorMap() happens to yield. Only the thrown exception is safe to assert on
    // - whether the aspect's AnalysisFailureEvent got posted first is order dependent.
    Label aspectLabel = Label.parseCanonicalUnchecked("//aspect_err");
    TopLevelAspectsKey aspectKey = topLevelAspectsKey(aspectLabel);
    ConfiguredTargetKey executionKey = configuredTargetKey("//exec_err");

    EvaluationResult<SkyValue> result =
        EvaluationResult.<SkyValue>builder()
            .addError(aspectKey, errorInfo(analysisException("aspect exception", aspectLabel)))
            .addError(
                executionKey,
                errorInfo(
                    actionExecutionExceptionWithAction(
                        "action failed",
                        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE))))
            .build();

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ true));

    assertThat(thrown).hasMessageThat().isEqualTo("TestAction failed: action failed");
  }

  // -------------------------------------------------------------------------------------------
  // F. Action conflicts.
  // -------------------------------------------------------------------------------------------

  @Test
  public void topLevelConflictException_keepGoing_collectsAllTransitiveConflicts()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//conflict");
    ActionAnalysisMetadata firstAction = mock(ActionAnalysisMetadata.class);
    ActionAnalysisMetadata secondAction = mock(ActionAnalysisMetadata.class);
    ActionConflictException firstConflict = actionConflictException("first", firstAction);
    ActionConflictException secondConflict = actionConflictException("second", secondAction);

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    new TopLevelConflictException(
                        "conflicts",
                        ImmutableMap.of(
                            firstAction, firstConflict, secondAction, secondConflict)))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.actionConflicts())
        .containsExactly(firstAction, firstConflict, secondAction, secondConflict);
    // Conflicts are analysis errors.
    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.aspectKeysForConflictReporting()).isEmpty();
  }

  @Test
  public void topLevelConflictException_noKeepGoing_doesNotThrow() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//conflict");
    ActionAnalysisMetadata action = mock(ActionAnalysisMetadata.class);
    ActionConflictException conflict = actionConflictException("conflict", action);

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    new TopLevelConflictException("conflicts", ImmutableMap.of(action, conflict)))),
            /* keepGoing= */ false,
            /* includeExecutionPhase= */ true);

    assertThat(result.actionConflicts()).containsExactly(action, conflict);
  }

  @Test
  public void actionConflictException_onConfiguredTargetKey_singleEntryAndNoAspectKeys()
      throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//conflict");
    ActionAnalysisMetadata action = mock(ActionAnalysisMetadata.class);
    ActionConflictException conflict = actionConflictException("conflict", action);

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(conflict)),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.actionConflicts()).containsExactly(action, conflict);
    assertThat(result.aspectKeysForConflictReporting()).isEmpty();
  }

  @Test
  public void actionConflictException_withAspectKeyInfo_reportsTheAspectKey() throws Exception {
    ConfiguredTargetKey baseKey = configuredTargetKey("//conflict");
    AspectKey aspectKey = aspectKey(baseKey);
    ActionAnalysisMetadata action = mock(ActionAnalysisMetadata.class);
    ActionConflictException conflict =
        ActionConflictException.withAspectKeyInfo(
            actionConflictException("conflict", action), aspectKey);

    ErrorProcessingResult result =
        processErrors(
            resultOf(aspectKey, errorInfo(conflict)),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.actionConflicts()).containsExactly(action, conflict);
    assertThat(result.aspectKeysForConflictReporting()).containsExactly(aspectKey);
  }

  @Test
  public void twoConflictsSharingTheSameActionKey_lastWinsAndDoesNotThrow() throws Exception {
    // The aggregating builder uses HashMap#putAll, so duplicate keys silently overwrite instead of
    // throwing (as ImmutableMap.Builder#buildOrThrow would).
    ConfiguredTargetKey firstKey = configuredTargetKey("//conflict_a");
    ConfiguredTargetKey secondKey = configuredTargetKey("//conflict_b");
    ActionAnalysisMetadata sharedAction = mock(ActionAnalysisMetadata.class);
    ActionConflictException firstConflict = actionConflictException("first", sharedAction);
    ActionConflictException secondConflict = actionConflictException("second", sharedAction);

    ErrorProcessingResult result =
        processErrors(
            EvaluationResult.<SkyValue>builder()
                .addError(firstKey, errorInfo(firstConflict))
                .addError(secondKey, errorInfo(secondConflict))
                .build(),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.actionConflicts()).hasSize(1);
    // ORDER DEPENDENCE: which of the two conflicts survives depends on errorMap() iteration order.
    assertThat(result.actionConflicts().get(sharedAction)).isAnyOf(firstConflict, secondConflict);
  }

  // -------------------------------------------------------------------------------------------
  // G. keepGoing = true aggregation across several errors.
  // -------------------------------------------------------------------------------------------

  @Test
  public void keepGoing_multipleExecutionErrors_keepsTheMostImportantExitCode() throws Exception {
    // DetailedExitCodeComparator ranks infrastructure failures above non-infrastructure ones, so
    // this is deterministic despite the HashMap iteration order.
    ConfiguredTargetKey infrastructureKey = configuredTargetKey("//infra_err");
    ConfiguredTargetKey buildKey = configuredTargetKey("//build_err");
    DetailedExitCode infrastructureExitCode =
        executionExitCode("infra failure", Execution.Code.EXECUTION_LOG_WRITE_FAILURE);
    DetailedExitCode buildExitCode =
        executionExitCode("build failure", Execution.Code.ACTION_NOT_UP_TO_DATE);

    ErrorProcessingResult result =
        processErrors(
            EvaluationResult.<SkyValue>builder()
                .addError(
                    infrastructureKey,
                    errorInfo(actionExecutionException("infra failure", infrastructureExitCode)))
                .addError(
                    buildKey, errorInfo(actionExecutionException("build failure", buildExitCode)))
                .build(),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(infrastructureExitCode);
  }

  @Test
  public void keepGoing_loadingAnalysisExecutionAndConflictErrors_allFieldsPopulated()
      throws Exception {
    ConfiguredTargetKey loadingKey = configuredTargetKey("//loading_err");
    ConfiguredTargetKey analysisKey = configuredTargetKey("//analysis_err");
    ConfiguredTargetKey executionKey = configuredTargetKey("//exec_err");
    ConfiguredTargetKey conflictKey = configuredTargetKey("//conflict");
    DetailedExitCode executionExitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);
    ActionAnalysisMetadata action = mock(ActionAnalysisMetadata.class);
    ActionConflictException conflict = actionConflictException("conflict", action);

    ErrorProcessingResult result =
        processErrors(
            EvaluationResult.<SkyValue>builder()
                .addError(
                    loadingKey,
                    errorInfo(
                        analysisExceptionWithCauses(
                            "loading exception",
                            loadingKey.getLabel(),
                            NestedSetBuilder.create(
                                Order.STABLE_ORDER,
                                new LoadingFailedCause(
                                    Label.parseCanonicalUnchecked("//missing"),
                                    analysisExitCode("missing"))))))
                .addError(
                    analysisKey,
                    errorInfo(analysisException("analysis exception", analysisKey.getLabel())))
                .addError(
                    executionKey,
                    errorInfo(actionExecutionException("action failed", executionExitCode)))
                .addError(conflictKey, errorInfo(conflict))
                .build(),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.hasLoadingError()).isTrue();
    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isEqualTo(executionExitCode);
    assertThat(result.actionConflicts()).containsExactly(action, conflict);
  }

  @Test
  public void keepGoing_nonExecutionError_emitsWarningWithCauseMessage() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//pkg:analysis_err");

    processErrors(
        resultOf(key, errorInfo(analysisException("the detailed reason", key.getLabel()))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ false);

    // SkymeldBuildIntegrationTest#testKeepGoingWarningContainsDetails asserts on this text, both
    // the first line and the appended cause message.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:analysis_err', it will not be"
                + " built.\n"
                + "the detailed reason");
  }

  @Test
  public void keepGoing_executionError_emitsNoWarning() throws Exception {
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");

    processErrors(
        resultOf(
            key,
            errorInfo(
                actionExecutionException(
                    "action failed",
                    executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE)))),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ true);

    assertThat(warningMessages()).isEmpty();
  }

  @Test
  public void keepGoing_oneWarningPerNonExecutionError() throws Exception {
    ConfiguredTargetKey firstKey = configuredTargetKey("//pkg:analysis_err_a");
    ConfiguredTargetKey secondKey = configuredTargetKey("//pkg:analysis_err_b");
    ConfiguredTargetKey executionKey = configuredTargetKey("//exec_err");

    processErrors(
        EvaluationResult.<SkyValue>builder()
            .addError(firstKey, errorInfo(analysisException("reason a", firstKey.getLabel())))
            .addError(secondKey, errorInfo(analysisException("reason b", secondKey.getLabel())))
            .addError(
                executionKey,
                errorInfo(
                    actionExecutionException(
                        "action failed",
                        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE))))
            .build(),
        /* keepGoing= */ true,
        /* includeExecutionPhase= */ true);

    // Order-independent on purpose: the warnings are emitted in errorMap() iteration order.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:analysis_err_a', it will not be"
                + " built.\n"
                + "reason a",
            "errors encountered while analyzing target '//pkg:analysis_err_b', it will not be"
                + " built.\n"
                + "reason b");
  }

  // -------------------------------------------------------------------------------------------
  // H. Entry point wrappers.
  // -------------------------------------------------------------------------------------------

  @Test
  public void processAnalysisErrors_executionError_throwsIllegalState() {
    // An execution *cycle* is used rather than an execution exception: an execution exception would
    // first trip the "this is not a valid analysis exception" bug report.
    ConfiguredTargetKey key = configuredTargetKey("//exec_cycle");

    EvaluationResult<SkyValue> result = resultOf(key, ErrorInfo.fromCycle(executionCycle(key)));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () ->
                SkyframeErrorProcessor.processAnalysisErrors(
                    result,
                    cyclesReporter,
                    eventHandler,
                    /* keepGoing= */ false,
                    /* keepEdges= */ false,
                    eventBus,
                    bugReporter));

    assertThat(thrown).hasMessageThat().contains("Unexpected execution phase exception");
    assertThat(thrown).hasCauseThat().isInstanceOf(BuildFailedException.class);
  }

  @Test
  public void processExecutionErrors_analysisError_throwsIllegalState() {
    ConfiguredTargetKey key = configuredTargetKey("//analysis_err");

    EvaluationResult<SkyValue> result =
        resultOf(key, errorInfo(analysisException("analysis exception", key.getLabel())));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () ->
                SkyframeErrorProcessor.processExecutionErrors(
                    result,
                    cyclesReporter,
                    eventHandler,
                    /* keepGoing= */ false,
                    /* keepEdges= */ false,
                    eventBus,
                    bugReporter));

    assertThat(thrown).hasMessageThat().contains("Unexpected analysis phase exception");
    assertThat(thrown).hasCauseThat().isInstanceOf(ViewCreationFailedException.class);
  }

  // -------------------------------------------------------------------------------------------
  // Pre-existing test, kept as-is (it exercises keepEdges = true).
  // -------------------------------------------------------------------------------------------

  @Test
  public void testProcessErrors_analysisErrorNoKeepGoing_throwsException(
      @TestParameter boolean includeExecutionPhase) throws Exception {
    ConfiguredTargetKey analysisErrorKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//analysis_err"))
            .build();
    TargetAndConfiguration mockTargetAndConfiguration =
        new TargetAndConfiguration(mock(Target.class), /* configuration= */ null);
    ConfiguredValueCreationException analysisException =
        new ConfiguredValueCreationException(
            mockTargetAndConfiguration.getTarget(), "analysis exception");
    ErrorInfo analysisErrorInfo =
        ErrorInfo.fromException(
            new ReifiedSkyFunctionException(
                new DummySkyFunctionException(analysisException, Transience.PERSISTENT)),
            /* isTransitivelyTransient= */ false);

    EvaluationResult<SkyValue> result =
        EvaluationResult.builder().addError(analysisErrorKey, analysisErrorInfo).build();

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                SkyframeErrorProcessor.processErrors(
                    result,
                    /* cyclesReporter= */ new CyclesReporter(),
                    /* eventHandler= */ mock(ExtendedEventHandler.class),
                    /* keepGoing= */ false,
                    /* keepEdges= */ true,
                    /* eventBus= */ null,
                    /* bugReporter= */ null,
                    includeExecutionPhase));
    assertThat(thrown).hasCauseThat().isEqualTo(analysisException);
  }

  // -------------------------------------------------------------------------------------------
  // I. Failure details and the less-travelled exception types.
  // -------------------------------------------------------------------------------------------

  @Test
  public void noSuchPackageException_keepGoing_analysisErrorWithPackageMissingRootCause()
      throws Exception {
    // convertToAnalysisException names NoSuchPackageException explicitly, but only its sibling
    // NoSuchTargetException was covered. Both take the NoSuchThingException arm of
    // processIndividualError; the observable difference is the exit code each one defaults to.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:missing_pkg_dep");
    NoSuchPackageException cause =
        new NoSuchPackageException(
            PackageIdentifier.createInMainRepo("missing_pkg"), "no such package");

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(cause)),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(result.executionDetailedExitCode()).isNull();

    ImmutableList<DetailedExitCode> rootCauseExitCodes =
        rootCauseDetailedExitCodes(onlyAnalysisFailureEvent());
    assertThat(rootCauseExitCodes).containsExactly(cause.getDetailedExitCode());
    FailureDetail rootCauseDetail = rootCauseExitCodes.get(0).getFailureDetail();
    // PACKAGE_MISSING, where the sibling NoSuchTargetException gives TARGET_MISSING.
    assertThat(rootCauseDetail.getPackageLoading().getCode())
        .isEqualTo(PackageLoading.Code.PACKAGE_MISSING);
    // Note: unlike a Label, a main-repo PackageIdentifier renders without the leading "//".
    assertThat(rootCauseDetail.getMessage())
        .isEqualTo("no such package 'missing_pkg': no such package");
  }

  @Test
  public void noKeepGoing_noSuchTargetException_failureDetailKeepsPackageLoadingSubfield() {
    // The NoSuchThingException arm carries a comment saying it exists for --nokeep_going, yet only
    // --keep_going was covered. This also pins maybeContextualizeFailureDetail: the message gets
    // the "Analysis of target ... failed; build aborted" prefix, while the original failure
    // detail's subfield and code survive - here a *non-analysis* subfield on a
    // ViewCreationFailedException, which is the interesting part.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:no_such_target");

    ViewCreationFailedException thrown =
        noKeepGoingAnalysisFailure(
            key, new NoSuchTargetException(key.getLabel(), "target doesn't exist"));

    assertThat(thrown.getFailureDetail().getMessage())
        .isEqualTo(
            "Analysis of target '//pkg:no_such_target' failed; build aborted: no such target"
                + " '//pkg:no_such_target': target doesn't exist");
    assertThat(thrown.getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(PackageLoading.Code.TARGET_MISSING);
    assertThat(thrown.getFailureDetail().hasAnalysis()).isFalse();
    // The 3-arg ViewCreationFailedException constructor concatenates, so the exception's own
    // message ends up identical to the contextualized failure detail message.
    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo(
            "Analysis of target '//pkg:no_such_target' failed; build aborted: no such target"
                + " '//pkg:no_such_target': target doesn't exist");
  }

  @Test
  public void noKeepGoing_noSuchPackageException_failureDetailKeepsPackageMissingCode() {
    // Same arm as above, different default code: the contextualization is agnostic to which
    // NoSuchThingException subclass it is handed.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:missing_pkg_dep");

    ViewCreationFailedException thrown =
        noKeepGoingAnalysisFailure(
            key,
            new NoSuchPackageException(
                PackageIdentifier.createInMainRepo("missing_pkg"), "no such package"));

    assertThat(thrown.getFailureDetail().getMessage())
        .isEqualTo(
            "Analysis of target '//pkg:missing_pkg_dep' failed; build aborted: no such package"
                + " 'missing_pkg': no such package");
    assertThat(thrown.getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(PackageLoading.Code.PACKAGE_MISSING);
    assertThat(thrown.getFailureDetail().hasAnalysis()).isFalse();
  }

  @Test
  public void noKeepGoing_targetCompatibilityCheckException_failureDetailKeepsAnalysisSubfield() {
    // The --nokeep_going half of the compatibility path, and the counterpart to the other
    // contextualization tests here: this failure detail *is* an analysis one, so hasAnalysis()
    // stays true and the code survives untouched. Worth pinning on its own, because the
    // UNEXPECTED_ANALYSIS_EXCEPTION fallback synthesizes an analysis-coded detail too - the code
    // is the only thing that tells a preserved detail from a synthesized one.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:incompatible");

    ViewCreationFailedException thrown =
        noKeepGoingAnalysisFailure(
            key,
            new TargetCompatibilityCheckException(
                "incompatible target", analysisFailureDetail("incompatible target")));

    assertThat(thrown.getFailureDetail().getMessage())
        .isEqualTo(
            "Analysis of target '//pkg:incompatible' failed; build aborted: incompatible target");
    assertThat(thrown.getFailureDetail().hasAnalysis()).isTrue();
    assertThat(thrown.getFailureDetail().getAnalysis().getCode())
        .isEqualTo(Analysis.Code.ANALYSIS_UNKNOWN);
  }

  @Test
  public void noKeepGoing_externalDepsException_failureDetailKeepsExternalDepsSubfield() {
    // The ExternalDepsException arm likewise only existed for --nokeep_going in the source
    // comments, and was only ever exercised with --keep_going.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:external_deps_err");

    ViewCreationFailedException thrown =
        noKeepGoingAnalysisFailure(
            key, ExternalDepsException.withMessage(ExternalDeps.Code.BAD_MODULE, "bad module"));

    assertThat(thrown.getFailureDetail().getMessage())
        .isEqualTo(
            "Analysis of target '//pkg:external_deps_err' failed; build aborted: bad module");
    assertThat(thrown.getFailureDetail().getExternalDeps().getCode())
        .isEqualTo(ExternalDeps.Code.BAD_MODULE);
    assertThat(thrown.getFailureDetail().hasAnalysis()).isFalse();
  }

  @Test
  public void validAnalysisExceptionOfUnrecognizedSubtype_filesBugReportFromIndividualProcessing() {
    // The fallback else branch of processIndividualError. Unlike
    // unrecognizedExceptionType_crashesWithABugReport, the exception here *is* a
    // SaneAnalysisException, so it passes validation and the bug report is filed from
    // processIndividualError rather than from assertValidAnalysisException.
    //
    // SYNTHETIC INPUT: production never gets here. ConfiguredTargetFunction wraps analysis failures
    // into a ConfiguredValueCreationException, and aspect exceptions ride on aspect keys and take
    // the aspect branch. That is precisely why the branch files a bug report: it is a "should never
    // happen" fallback. Because it goes through the *static* BugReport, the test cannot observe a
    // result and has to assert on the thrown IllegalStateException instead.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:aspect_creation_err");

    EvaluationResult<SkyValue> result =
        resultOf(
            key, errorInfo(new AspectCreationException("aspect creation failed", key.getLabel())));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> processErrors(result, /* keepGoing= */ true, /* includeExecutionPhase= */ false));

    assertThat(thrown).hasMessageThat().contains("Unexpected cause encountered while evaluating");
  }

  /** Runs an analysis-phase {@code --nokeep_going} {@code processErrors} that must throw. */
  private ViewCreationFailedException noKeepGoingAnalysisFailure(
      ConfiguredTargetKey key, Exception cause) {
    return assertThrows(
        ViewCreationFailedException.class,
        () ->
            processErrors(
                resultOf(key, errorInfo(cause)),
                /* keepGoing= */ false,
                /* includeExecutionPhase= */ false));
  }

  // -------------------------------------------------------------------------------------------
  // J. Aspect error paths other than a plain analysis exception.
  // -------------------------------------------------------------------------------------------

  @Test
  public void aspectKey_topLevelConflictException_keepGoing_collectsConflictsAndPostsNoEvent()
      throws Exception {
    // Pins the TopLevelConflictException arm of the aspect branch: the transitive conflicts are
    // collected exactly as they are for a ConfiguredTargetKey, the error counts as an analysis
    // error, and - because it is an action conflict - nothing is posted and nothing is warned
    // about, it is all deferred to SkyframeBuildView.
    // Wart: aspectKeysForConflictReporting stays empty even though the failing key *is* an aspect
    // key; it is only ever populated from ActionConflictException#getAspectKey.
    TopLevelAspectsKey key =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_conflict"));
    ActionAnalysisMetadata action = mock(ActionAnalysisMetadata.class);
    ActionConflictException conflict = actionConflictException("conflict", action);

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    new TopLevelConflictException("conflicts", ImmutableMap.of(action, conflict)))),
            /* keepGoing= */ true,
            // Required: assertValidAnalysisException does not accept a TopLevelConflictException,
            // only assertValidAnalysisOrExecutionException does.
            /* includeExecutionPhase= */ true);

    assertThat(result.actionConflicts()).containsExactly(action, conflict);
    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(result.aspectKeysForConflictReporting()).isEmpty();
    assertThat(eventBusCollector.allEvents).isEmpty();
    assertThat(warningMessages()).isEmpty();
  }

  @Test
  public void topLevelAspectsKey_executionException_keepGoing_noAnalysisFailureEvent()
      throws Exception {
    // Pins the isExecutionException arm of the aspect branch: the exit code is taken from the
    // exception, and since the error is then not an analysis error, *no* AnalysisFailureEvent is
    // posted - unlike an aspect analysis error, which posts one for its base configured target.
    TopLevelAspectsKey key =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_exec_err"));
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    ErrorProcessingResult result =
        processErrors(
            resultOf(key, errorInfo(actionExecutionExceptionWithAction("action failed", exitCode))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(exitCode);
    assertThat(result.hasAnalysisError()).isFalse();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(eventBusCollector.analysisFailures).isEmpty();
    assertThat(eventBusCollector.allEvents).isEmpty();
  }

  // TODO(b/561978611): Remove this behavior. An execution cycle on an aspect is reported with
  // analysis wording, because its cause is null.
  @Test
  public void topLevelAspectsKey_executionCycle_keepGoing_cycleCodeAndNoAnalysisFailureEvent()
      throws Exception {
    // Pins the execution-cycle arm of the aspect branch: CYCLE_CODE, not an analysis error, and
    // again no AnalysisFailureEvent.
    TopLevelAspectsKey key =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_exec_cycle"));

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                ErrorInfo.fromCycle(
                    executionCycle(configuredTargetKey("//pkg:aspect_exec_cycle")))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.executionDetailedExitCode()).isEqualTo(EXECUTION_CYCLE_CODE);
    assertThat(result.hasAnalysisError()).isFalse();
    assertThat(eventBusCollector.analysisFailures).isEmpty();
    assertThat(eventBusCollector.allEvents).isEmpty();
    // Wart: a cycle carries no exception, so isExecutionException(null) is false and the keep_going
    // warning claims this was an *analysis* failure, even though the error was just classified as
    // an execution one. The message also has no cause line, because there is no cause.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:aspect_exec_cycle', it will not be"
                + " built.");
  }

  // TODO(b/561978611): Remove this behavior. An aspect's loading causes are dropped, so
  // hasLoadingError() can never become true.
  @Test
  public void topLevelAspectsKey_loadingFailedCause_keepGoing_noLoadingErrorAndNoEvent()
      throws Exception {
    // Pins a wart: the aspect branch hardcodes an empty loading-root-cause set, so a
    // LoadingFailedCause carried by the exception is dropped - hasLoadingError() is false and no
    // LoadingFailureEvent is posted. The very same exception on a ConfiguredTargetKey reports both
    // (see loadingError_postsOneLoadingFailureEventPerDedupedLabel).
    Label label = Label.parseCanonicalUnchecked("//pkg:aspect_loading_err");
    TopLevelAspectsKey key = topLevelAspectsKey(label);
    Label loadingRootCause = Label.parseCanonicalUnchecked("//pkg:missing_dep");

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    analysisExceptionWithCauses(
                        "aspect loading exception",
                        label,
                        NestedSetBuilder.create(
                            Order.STABLE_ORDER,
                            new LoadingFailedCause(
                                loadingRootCause, analysisExitCode("missing")))))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasLoadingError()).isFalse();
    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(eventBusCollector.loadingFailures).isEmpty();
    // The AnalysisFailureEvent is still posted, but with the root causes discarded.
    assertThat(onlyAnalysisFailureEvent().getRootCauses().toList()).isEmpty();
  }

  @Test
  public void noKeepGoing_topLevelAspectsKeyExecutionError_throwsImmediatelyWithoutDeferral() {
    // Pins the ordering inside throwOrReturnAspectAnalysisException: the isExecutionException check
    // comes *before* the TopLevelAspectsKey check, so an execution failure on a top-level aspect is
    // rethrown on the spot instead of being stashed and rethrown after the loop, which is what
    // happens to an aspect *analysis* error (see
    // noKeepGoing_aspectErrorPlusTargetAnalysisError_targetErrorWins).
    TopLevelAspectsKey key =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_exec_err"));
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        errorInfo(actionExecutionExceptionWithAction("action failed", exitCode))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown.getDetailedExitCode()).isEqualTo(exitCode);
    assertThat(thrown).hasMessageThat().isEqualTo("TestAction failed: action failed");
    assertThat(eventBusCollector.analysisFailures).isEmpty();
  }

  @Test
  public void noKeepGoing_topLevelAspectsKeyExecutionCycle_throwsBuildFailedWithCycleCode() {
    // Same bypass as above, through the hasExecutionCycle check: no deferral, no
    // ViewCreationFailedException, just the generic cycle BuildFailedException with a null message.
    TopLevelAspectsKey key =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_exec_cycle"));

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        ErrorInfo.fromCycle(
                            executionCycle(configuredTargetKey("//pkg:aspect_exec_cycle")))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown.getDetailedExitCode()).isEqualTo(EXECUTION_CYCLE_CODE);
    assertThat(thrown).hasMessageThat().isNull();
    assertThat(eventBusCollector.analysisFailures).isEmpty();
  }

  // TODO(b/561978611): Remove this behavior. Which aspect error aborts the build depends on hash
  // order, so the same build can fail two different ways.
  @Test
  public void noKeepGoing_twoTopLevelAspectsAnalysisErrors_postsBothEventsAndThrowsAfterTheLoop() {
    // Pins the contrast with two ConfiguredTargetKey analysis errors, where the first one reached
    // throws immediately and no further events are posted: the aspect deferral path *returns* the
    // exception instead of throwing, so the loop runs to completion, an AnalysisFailureEvent is
    // posted for every aspect, and the stashed exception is overwritten on each iteration before
    // being thrown after the loop.
    //
    // ORDER DEPENDENCE: deliberately not pinning *which* of the two is thrown. EvaluationResult
    // stores errors in a HashMap, and TopLevelAspectsKey's hash folds in
    // ImmutableList<AspectClass>#hashCode, where ASPECT_CLASS is a lambda that does not override
    // hashCode - so it gets an identity hash and the iteration order flips between JVM runs.
    Label firstLabel = Label.parseCanonicalUnchecked("//pkg:aspect_err_a");
    Label secondLabel = Label.parseCanonicalUnchecked("//pkg:aspect_err_b");
    TopLevelAspectsKey firstKey = topLevelAspectsKey(firstLabel);
    TopLevelAspectsKey secondKey = topLevelAspectsKey(secondLabel);
    ConfiguredValueCreationException firstCause =
        analysisException("aspect exception a", firstLabel);
    ConfiguredValueCreationException secondCause =
        analysisException("aspect exception b", secondLabel);

    EvaluationResult<SkyValue> result =
        EvaluationResult.<SkyValue>builder()
            .addError(firstKey, errorInfo(firstCause))
            .addError(secondKey, errorInfo(secondCause))
            .build();

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ false));

    assertThat(analysisFailureTargets())
        .containsExactly(
            firstKey.getBaseConfiguredTargetKey(), secondKey.getBaseConfiguredTargetKey());
    assertThat(thrown).hasCauseThat().isAnyOf(firstCause, secondCause);
  }

  // TODO(b/561978611): Remove this behavior. An aspect failure should carry its root causes, the
  // way a target failure does.
  @Test
  public void aspectAnalysisCycle_keepGoing_analysisErrorWithEmptyRootCauses() throws Exception {
    // Pins an analysis (i.e. non-execution) cycle on a top-level aspect: an analysis error with no
    // execution exit code, and an AnalysisFailureEvent for the base configured target.
    // Wart: the aspect branch never runs the cycle-culprit helper, so the event has *empty* root
    // causes, whereas the same cycle on a ConfiguredTargetKey synthesizes a LabelCause for the
    // culprit (see analysisCycle_keepGoing_analysisErrorWithCycleLabelCause).
    TopLevelAspectsKey key =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_cycle"));
    ConfiguredTargetKey culprit = configuredTargetKey("//pkg:culprit");

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key, ErrorInfo.fromCycle(CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
    AnalysisFailureEvent event = onlyAnalysisFailureEvent();
    assertThat(event.getFailedTarget()).isEqualTo(key.getBaseConfiguredTargetKey());
    assertThat(event.getRootCauses().toList()).isEmpty();
  }

  // TODO(b/561978611): Remove this behavior. An aspect failure should carry its root causes, the
  // way a target failure does.
  @Test
  public void aspectAnalysisCycle_noKeepGoing_throwsViewCreationFailedWithCycleCode() {
    // Pins the exact aspect cycle failure detail. The description comes from
    // TopLevelAspectsKey#getDescription: the aspect class names, the (empty) parameters map and the
    // target label.
    TopLevelAspectsKey key =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_cycle"));
    ConfiguredTargetKey culprit = configuredTargetKey("//pkg:culprit");

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        ErrorInfo.fromCycle(CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ false));

    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo(
            "Analysis of aspects '[TestAspect] with parameters {} on //pkg:aspect_cycle' failed;"
                + " build aborted");
    // A cycle has no exception, so the failure detail message gets the " due to cycle" suffix and
    // the exception has no cause.
    assertThat(thrown.getFailureDetail().getMessage())
        .isEqualTo(
            "Analysis of aspects '[TestAspect] with parameters {} on //pkg:aspect_cycle' failed;"
                + " build aborted due to cycle");
    assertThat(thrown.getFailureDetail().getAnalysis().getCode()).isEqualTo(Analysis.Code.CYCLE);
    assertThat(thrown).hasCauseThat().isNull();
    // Same wart as in keep_going mode: the event is posted before the throw, with no root causes.
    assertThat(onlyAnalysisFailureEvent().getRootCauses().toList()).isEmpty();
  }

  // TODO(b/561978611): Remove this behavior. An aspect failure should carry its root causes, the
  // way a target failure does.
  @Test
  public void buildDriverKeyWrappingTopLevelAspectsKey_isHandledLikeTheBareKey() throws Exception {
    // This is the real Skymeld top-level-aspect shape. getEffectiveErrorKey only calls
    // getActionLookupKey() and ignores isTopLevelAspectDriver, so it behaves exactly like the bare
    // TopLevelAspectsKey (see
    // topLevelAspectsKey_postsAnalysisFailureEventForBaseTargetWithEmptyRootCauses).
    TopLevelAspectsKey aspectsKey =
        topLevelAspectsKey(Label.parseCanonicalUnchecked("//pkg:aspect_err"));
    SkyKey key =
        BuildDriverKey.ofTopLevelAspect(
            aspectsKey,
            TOP_LEVEL_ARTIFACT_CONTEXT,
            /* explicitlyRequested= */ true,
            /* skipIncompatibleExplicitTargets= */ false,
            /* extraActionTopLevelOnly= */ false,
            /* keepGoing= */ true);

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(analysisException("aspect analysis exception", aspectsKey.getLabel()))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
    AnalysisFailureEvent event = onlyAnalysisFailureEvent();
    assertThat(event.getFailedTarget()).isEqualTo(aspectsKey.getBaseConfiguredTargetKey());
    assertThat(event.getRootCauses().toList()).isEmpty();
  }

  // TODO(b/561978611): Remove this behavior. Failing a build with a raw ClassCastException tells
  // the user nothing.
  @Test
  public void noKeepGoing_aspectCompletionKey_throwsClassCastException() {
    // An AspectCompletionKey unwraps to a *bare* AspectKey, not a TopLevelAspectsKey, so the
    // deferral check does not match and the key falls through to the ConfiguredTargetKey cast: the
    // same wart as noKeepGoing_bareAspectKeyAnalysisError_throwsClassCastException, reached here
    // through the wrapper. The other wrapper tests only ever run with keepGoing = true.
    ConfiguredTargetKey baseKey = configuredTargetKey("//pkg:aspect_err");
    SkyKey key = wrapKey(TopLevelKeyKind.ASPECT_COMPLETION, baseKey);

    EvaluationResult<SkyValue> result =
        resultOf(key, errorInfo(analysisException("aspect exception", baseKey.getLabel())));

    assertThrows(
        ClassCastException.class,
        () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ false));
  }

  /** The base configured targets of every posted {@link AnalysisFailureEvent}. */
  private ImmutableList<ConfiguredTargetKey> analysisFailureTargets() {
    return eventBusCollector.analysisFailures.stream()
        .map(AnalysisFailureEvent::getFailedTarget)
        .collect(toImmutableList());
  }

  // -------------------------------------------------------------------------------------------
  // K. ActionLookupData warts, cycle culprits, and exception-plus-cycle precedence.
  // -------------------------------------------------------------------------------------------

  // TODO(b/561978611): Remove this behavior. Failing a build with a raw ClassCastException tells
  // the user nothing.
  @Test
  public void noKeepGoing_actionLookupDataAnalysisError_throwsClassCastException() {
    // Wart: an ActionLookupData carrying a *non-execution* exception has isExecutionException ==
    // false, so --nokeep_going enters throwOrReturnAspectAnalysisException, which only
    // special-cases TopLevelAspectsKey and then casts the key to ConfiguredTargetKey. This is the
    // same cast as in noKeepGoing_bareAspectKeyAnalysisError_throwsClassCastException above, for a
    // different key type.
    ConfiguredTargetKey ctKey = configuredTargetKey("//pkg:build_info");
    ActionLookupData key = ActionLookupData.create(ctKey, /* actionIndex= */ 0);
    EvaluationResult<SkyValue> result =
        resultOf(key, errorInfo(analysisException("analysis exception", ctKey.getLabel())));

    // The exception type only: the ClassCastException message is JVM-version-dependent.
    assertThrows(
        ClassCastException.class,
        () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ true));
  }

  // TODO(b/561978611): Remove this behavior. A cycle on an ActionLookupData crashes with a
  // NullPointerException.
  @Test
  public void actionLookupDataCycle_filesBugReportThenThrowsNullPointerException(
      @TestParameter boolean keepGoing) {
    // Latent bug, in both keep_going modes: for an ActionLookupData key, processIndividualError
    // unconditionally takes the execution path and hands the ErrorInfo's exception - null, for a
    // cycle - to getExecutionDetailedExitCodeFromCause. DetailedException.getDetailedExitCode(null)
    // is harmless there (a plain instanceof check that returns null), so the null cause reaches
    // sendBugReportAndCreateUnknownExecutionDetailedExitCode, which files a non-fatal bug report
    // and *then* dereferences the cause at SkyframeErrorProcessor.java:630, i.e.
    // "Unexpected exception, please file an issue with the Bazel team: " + cause.getMessage().
    // That second call, not the getDetailedExitCode one, is where the NPE comes from. All of it
    // happens before the keepGoing branch, hence the identical crash in both modes.
    ConfiguredTargetKey ctKey = configuredTargetKey("//pkg:build_info");
    ActionLookupData key = ActionLookupData.create(ctKey, /* actionIndex= */ 0);
    // The cycle's contents do not matter: the ActionLookupData branch never inspects them, so not
    // even an execution cycle gets the CYCLE_CODE shortcut that the other key types get.
    EvaluationResult<SkyValue> result = resultOf(key, ErrorInfo.fromCycle(executionCycle(ctKey)));

    assertThrows(
        NullPointerException.class,
        () -> processErrors(result, keepGoing, /* includeExecutionPhase= */ true));

    // Filed on the injected bug reporter, before the NPE was thrown.
    assertThat(bugReporter.nonFatalBugReports).hasSize(1);
    assertThat(bugReporter.nonFatalBugReports.get(0))
        .hasMessageThat()
        .startsWith("action terminated with unexpected exception with result");
  }

  @Test
  public void analysisCycle_transitiveTargetKeyCulprit_rootCauseIsTheCulpritLabel()
      throws Exception {
    // Pins the TransitiveTargetKey.NAME branch of maybeGetConfiguredTargetCycleCulprit: the root
    // cause is the culprit's own label, not the top-level one. TransitiveTargetKey is also the one
    // key type here that overrides SkyKey#argument (it returns itself), which is why that branch
    // can cast the key directly while the CONFIGURED_TARGET branch casts argument().
    ConfiguredTargetKey key = configuredTargetKey("//pkg:cycle");
    TransitiveTargetKey culprit =
        TransitiveTargetKey.of(Label.parseCanonicalUnchecked("//cycle:transitive_culprit"));

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key, ErrorInfo.fromCycle(CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(onlyAnalysisFailureEvent().getRootCauses().toList())
        .containsExactly(dependencyCycleCause(culprit.getLabel()));
    // The cause is null for a cycle, so the warning has no appended cause message.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:cycle', it will not be built.");
  }

  @Test
  public void analysisCycle_otherCulpritKind_rootCauseFallsBackToTheTopLevelLabel()
      throws Exception {
    // Pins the else branch of maybeGetConfiguredTargetCycleCulprit: a culprit that is neither a
    // CONFIGURED_TARGET nor a TRANSITIVE_TARGET key - here an AspectKey, whose functionName is
    // SkyFunctions.ASPECT - makes the root cause the label of the *top-level* target, so the
    // reported cause names something that is not actually in the cycle.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:cycle");
    AspectKey culprit = aspectKey(configuredTargetKey("//other:base"));

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key, ErrorInfo.fromCycle(CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ false);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(onlyAnalysisFailureEvent().getRootCauses().toList())
        .containsExactly(dependencyCycleCause(key.getLabel()));
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:cycle', it will not be built.");
  }

  // TODO(b/561978611): Remove this behavior. The execution failure is silently dropped in favour of
  // the cycle.
  @Test
  public void executionExceptionPlusAnalysisCycle_keepGoing_cycleWinsAndDropsTheExecutionFailure()
      throws Exception {
    // Wart: on a ConfiguredTargetKey the cycle arm of processIndividualError is checked before the
    // isExecutionException arm, so an ErrorInfo carrying *both* an ActionExecutionException and an
    // analysis cycle is classified as a pure analysis cycle: the execution exit code is silently
    // dropped, and the caller never learns that an action failed.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:both");
    ConfiguredTargetKey culprit = configuredTargetKey("//cycle:culprit");
    ActionExecutionException cause =
        actionExecutionException(
            "action failed",
            executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE));

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key, errorInfo(key, cause, CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.hasAnalysisError()).isTrue();
    // The surprising part: the ActionExecutionException's exit code is gone.
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(onlyAnalysisFailureEvent().getRootCauses().toList())
        .containsExactly(dependencyCycleCause(culprit.getLabel()));
    // Also surprising: no warning either. logOrPrintWarningsKeepGoing is passed
    // isExecutionException(rawCause), which is true for the ActionExecutionException even though
    // the error was just classified as an analysis cycle, so it returns without warning.
    assertThat(warningMessages()).isEmpty();
  }

  @Test
  public void executionExceptionPlusAnalysisCycle_noKeepGoing_throwsBuildFailedException() {
    // The other half of the divergence: --nokeep_going consults isExecutionException on the raw
    // cause *before* looking at the classification, so the very same ErrorInfo that keep_going
    // reports as an analysis cycle is rethrown here as an execution failure.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:both");
    ConfiguredTargetKey culprit = configuredTargetKey("//cycle:culprit");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);
    EvaluationResult<SkyValue> result =
        resultOf(
            key,
            errorInfo(
                key,
                actionExecutionExceptionWithAction("action failed", exitCode),
                CycleInfo.createCycleInfo(ImmutableList.of(culprit))));

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ true));

    assertThat(thrown.getDetailedExitCode()).isEqualTo(exitCode);
    assertThat(thrown).hasMessageThat().isEqualTo("TestAction failed: action failed");
    // The AnalysisFailureEvent for the cycle is still posted before the throw.
    assertThat(eventBusCollector.analysisFailures).hasSize(1);
  }

  @Test
  public void analysisExceptionPlusAnalysisCycle_keepGoing_exceptionWinsOverTheCycle()
      throws Exception {
    // The other side of the precedence order, of which the two tests above only show one half: on
    // a ConfiguredTargetKey the TopLevelConflictException, ActionConflictException and
    // ConfiguredValueCreationException arms come *before* the cycle arm, while the cycle arm comes
    // before the NoSuchThingException, ExternalDepsException, TargetCompatibilityCheckException
    // and execution arms. So here the exception's root causes win and the cycle contributes
    // nothing at all.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:both");
    ConfiguredTargetKey culprit = configuredTargetKey("//cycle:culprit");
    LabelCause rootCause =
        new LabelCause(Label.parseCanonicalUnchecked("//pkg:dep"), analysisExitCode("dep failed"));

    ErrorProcessingResult result =
        processErrors(
            resultOf(
                key,
                errorInfo(
                    key,
                    analysisExceptionWithCauses(
                        "analysis exception",
                        key.getLabel(),
                        NestedSetBuilder.create(Order.STABLE_ORDER, rootCause)),
                    CycleInfo.createCycleInfo(ImmutableList.of(culprit)))),
            /* keepGoing= */ true,
            /* includeExecutionPhase= */ true);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
    // No dependencyCycleCause(//cycle:culprit) anywhere: the cycle is invisible in the output.
    assertThat(onlyAnalysisFailureEvent().getRootCauses().toList()).containsExactly(rootCause);
    // And, unlike the execution-exception case above, a warning *is* emitted for this one.
    assertThat(warningMessages())
        .containsExactly(
            "errors encountered while analyzing target '//pkg:both', it will not be built.\n"
                + "analysis exception");
  }

  /** The root cause that an analysis cycle synthesizes for its culprit. */
  private static LabelCause dependencyCycleCause(Label culpritLabel) {
    return new LabelCause(
        culpritLabel,
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage("Dependency cycle")
                .setAnalysis(Analysis.newBuilder().setCode(Analysis.Code.CYCLE))
                .build()));
  }

  // -------------------------------------------------------------------------------------------
  // L. rethrow() details, the graph walk, validation and the entry point wrappers.
  // -------------------------------------------------------------------------------------------

  @Test
  public void noKeepGoing_actionExecutionErrorWithLocation_messageIsPrefixedWithTheLocation() {
    // Pins the location prefix in rethrow(). Only the null-location side is pinned elsewhere, by
    // noKeepGoing_singleActionExecutionError_throwsBuildFailedException.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        errorInfo(
                            actionExecutionExceptionWithAction(
                                "action failed",
                                exitCode,
                                /* catastrophe= */ false,
                                Location.fromFileLineColumn("pkg/BUILD", 12, 3)))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    // The action description is applied first, the location second.
    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo("pkg/BUILD:12:3 TestAction failed: action failed");
  }

  @Test
  public void noKeepGoing_catastrophicActionExecutionError_buildFailedExceptionIsCatastrophic() {
    // Pins that rethrow() carries ActionExecutionException#isCatastrophe over to the
    // BuildFailedException.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        errorInfo(
                            actionExecutionExceptionWithAction(
                                "action failed",
                                exitCode,
                                /* catastrophe= */ true,
                                /* location= */ null))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown.isCatastrophic()).isTrue();
  }

  // TODO(b/561978611): Remove this behavior. rethrow only carries the catastrophe bit over for an
  // ActionExecutionException, so every other execution exception silently loses it.
  @Test
  public void noKeepGoing_catastrophicArtifactNestedSetEvalException_isNotCatastrophic() {
    // The contrast with the test above. An ArtifactNestedSetEvalException has its own
    // isCatastrophic flag, which ArtifactNestedSetFunction propagates up through the nested-set
    // evaluation - and which rethrow() then drops, because the exception reaches the final
    // "unexpected exception" fallback rather than the ActionExecutionException branch.
    ConfiguredTargetKey key = configuredTargetKey("//nested_set_err");

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        errorInfo(
                            artifactNestedSetEvalException(
                                "nested set failed", /* catastrophic= */ true))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown.isCatastrophic()).isFalse();
    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo(
            "Unexpected exception, please file an issue with the Bazel team: nested set failed");
    assertThat(thrown.getDetailedExitCode().getFailureDetail().getExecution().getCode())
        .isEqualTo(Execution.Code.UNEXPECTED_EXCEPTION);
    // Once while classifying the error and once again in rethrow().
    assertThat(bugReporter.nonFatalBugReports).hasSize(2);
  }

  // TODO(b/561978611): Remove this behavior. The rethrown message reads "TestAction failed: null".
  @Test
  public void noKeepGoing_actionExecutionErrorWithNoMessage_errorIsMarkedAlreadyShown() {
    // Pins the !showError() branch of rethrow(). The base showError() is getMessage() != null, so
    // with a plain ActionExecutionException the only way to reach it is a null message; the
    // subclass AlreadyReportedActionExecutionException hard-codes false instead, see
    // noKeepGoing_alreadyReportedActionExecutionError_errorIsMarkedAlreadyShown. Wart: the
    // rethrown message is *not* null - it comes out as "TestAction failed: null" - which is what
    // makes isErrorAlreadyShown() prove the showError() branch rather than BuildFailedException's
    // own null-message shortcut.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        errorInfo(
                            actionExecutionExceptionWithAction(
                                /* message= */ null,
                                exitCode,
                                /* catastrophe= */ false,
                                /* location= */ null))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown).hasMessageThat().isEqualTo("TestAction failed: null");
    assertThat(thrown.isErrorAlreadyShown()).isTrue();
  }

  @Test
  public void noKeepGoing_alreadyReportedActionExecutionError_errorIsMarkedAlreadyShown() {
    // The production-realistic route into the same branch: SkyframeActionExecutor and
    // ActionExecutionFunction wrap failures they have already reported in
    // AlreadyReportedActionExecutionException, the one showError() override in the codebase.
    // Unlike the null-message case above, the message survives intact, so this pins the
    // errorAlreadyShown mapping without also depending on the "TestAction failed: null" wart.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                processErrors(
                    resultOf(
                        key,
                        errorInfo(
                            new AlreadyReportedActionExecutionException(
                                actionExecutionExceptionWithAction("action failed", exitCode)))),
                    /* keepGoing= */ false,
                    /* includeExecutionPhase= */ true));

    assertThat(thrown).hasMessageThat().isEqualTo("TestAction failed: action failed");
    assertThat(thrown.isErrorAlreadyShown()).isTrue();
  }

  // TODO(b/561978611): Remove this behavior. rethrow dereferences the action's owner
  // unconditionally, so the getAction() null check guards nothing.
  @Test
  public void noKeepGoing_actionExecutionErrorWithOwnerlessAction_throwsNullPointerException() {
    // Latent bug, pinned on purpose: rethrow() guards on getAction() != null and its comment
    // claims to handle "Actions with no owner", but it then calls getLocation() unconditionally,
    // which dereferences action.getOwner(). An action with a null owner passes the guard, gets its
    // description prepended, and then NPEs.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    ActionAnalysisMetadata ownerlessAction = mock(ActionAnalysisMetadata.class);
    when(ownerlessAction.describe()).thenReturn("TestAction");
    when(ownerlessAction.getOwner()).thenReturn(null);
    ActionExecutionException cause =
        new ActionExecutionException(
            "action failed",
            ownerlessAction,
            /* catastrophe= */ false,
            executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE));

    EvaluationResult<SkyValue> result = resultOf(key, errorInfo(cause));

    assertThrows(
        NullPointerException.class,
        () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ true));
  }

  // TODO(b/561978611): Remove this behavior. rethrow dereferences the action's owner
  // unconditionally, so the getAction() null check guards nothing.
  @Test
  public void noKeepGoing_actionExecutionErrorWithNoAction_throwsNullPointerException() {
    // The same latent bug, other flavour: getLocation() also dereferences a *null* action, so the
    // getAction() != null guard protects nothing. This flavour is only reachable through rethrow()
    // with --nokeep_going: every other use of the null-action actionExecutionException helper is
    // --keep_going (rethrow is never called), and
    // noKeepGoing_testExecExceptionNestedInActionExecutionException_isRethrown escapes earlier via
    // the nested TestExecException.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");

    EvaluationResult<SkyValue> result =
        resultOf(
            key,
            errorInfo(
                actionExecutionException(
                    "action failed",
                    executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE))));

    assertThrows(
        NullPointerException.class,
        () -> processErrors(result, /* keepGoing= */ false, /* includeExecutionPhase= */ true));
  }

  @Test
  public void unrecognizedExceptionType_keepEdges_walksTheGraphToTheOriginOfTheException()
      throws Exception {
    // Pins the keepEdges = true half of logUnexpectedExceptionOrigin: the walk follows the direct
    // dep that threw the very same exception instance, and the traversed path is stringified into
    // the bug report message. The keepEdges = false half is pinned by
    // unrecognizedExceptionType_crashesWithABugReport.
    ConfiguredTargetKey key = configuredTargetKey("//unrecognized");
    ConfiguredTargetKey dep = configuredTargetKey("//unrecognized:dep");
    UnrecognizedException cause = new UnrecognizedException("not a known type");
    WalkableGraph walkableGraph = mock(WalkableGraph.class);
    when(walkableGraph.getMissingAndExceptions(ImmutableList.of(key)))
        .thenReturn(ImmutableMap.of());
    when(walkableGraph.getMissingAndExceptions(ImmutableList.of(dep)))
        .thenReturn(ImmutableMap.of());
    when(walkableGraph.getDirectDeps(key)).thenReturn(ImmutableList.of(dep));
    when(walkableGraph.getDirectDeps(dep)).thenReturn(ImmutableList.of());
    when(walkableGraph.getException(dep)).thenReturn(cause);

    EvaluationResult<SkyValue> result =
        EvaluationResult.<SkyValue>builder()
            .addError(key, errorInfo(cause))
            .setWalkableGraph(walkableGraph)
            .build();

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () ->
                processErrorsKeepingEdges(
                    result, /* keepGoing= */ true, /* includeExecutionPhase= */ false));

    assertThat(thrown).hasMessageThat().contains("Unexpected analysis error");
    assertThat(thrown).hasMessageThat().contains("(" + ImmutableList.of(key, dep) + ")");
  }

  @Test
  public void unrecognizedExceptionType_keepEdges_stopsWalkingAtAMissingNode() throws Exception {
    // Pins the break in logUnexpectedExceptionOrigin: when the missing map maps the current key to
    // null (a --nokeep_going build doesn't write bubbled-up error nodes to the graph), the walk
    // stops right there. The dep below *would* be followed otherwise, so the single-element path
    // is what proves the break.
    ConfiguredTargetKey key = configuredTargetKey("//unrecognized");
    ConfiguredTargetKey dep = configuredTargetKey("//unrecognized:dep");
    UnrecognizedException cause = new UnrecognizedException("not a known type");
    WalkableGraph walkableGraph = mock(WalkableGraph.class);
    when(walkableGraph.getMissingAndExceptions(ImmutableList.of(key)))
        .thenReturn(Collections.<SkyKey, Exception>singletonMap(key, null));
    when(walkableGraph.getDirectDeps(key)).thenReturn(ImmutableList.of(dep));
    when(walkableGraph.getException(dep)).thenReturn(cause);

    EvaluationResult<SkyValue> result =
        EvaluationResult.<SkyValue>builder()
            .addError(key, errorInfo(cause))
            .setWalkableGraph(walkableGraph)
            .build();

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () ->
                processErrorsKeepingEdges(
                    result, /* keepGoing= */ true, /* includeExecutionPhase= */ false));

    assertThat(thrown).hasMessageThat().contains("(" + ImmutableList.of(key) + ")");
  }

  // TODO(b/561978611): Remove this behavior. An unattributable action conflict is not a
  // programming error, so it should not crash with a bug report.
  @Test
  public void analysisOnly_topLevelConflictException_crashesWithABugReport() {
    // Pins an asymmetry: assertValidAnalysisOrExecutionException explicitly whitelists
    // TopLevelConflictException, assertValidAnalysisException does not - it is neither a
    // SaneAnalysisException nor a DetailedException. Every other TopLevelConflictException test in
    // this file passes includeExecutionPhase = true, which is why this case is uncovered.
    ConfiguredTargetKey key = configuredTargetKey("//conflict");
    ActionAnalysisMetadata action = mock(ActionAnalysisMetadata.class);
    ActionConflictException conflict = actionConflictException("conflict", action);

    EvaluationResult<SkyValue> result =
        resultOf(
            key,
            errorInfo(
                new TopLevelConflictException("conflicts", ImmutableMap.of(action, conflict))));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> processErrors(result, /* keepGoing= */ true, /* includeExecutionPhase= */ false));

    assertThat(thrown).hasMessageThat().contains("Unexpected analysis error");
    assertThat(thrown).hasMessageThat().contains("direct deps not stored");
  }

  @Test
  public void analysisOnly_executionException_crashesWithABugReport() {
    // Pins that an execution exception is not a valid *analysis* exception: with
    // includeExecutionPhase = false it trips the bug report before any classification happens.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");

    EvaluationResult<SkyValue> result =
        resultOf(
            key,
            errorInfo(
                actionExecutionException(
                    "action failed",
                    executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE))));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> processErrors(result, /* keepGoing= */ true, /* includeExecutionPhase= */ false));

    assertThat(thrown).hasMessageThat().contains("Unexpected analysis error");
    assertThat(thrown).hasMessageThat().contains("direct deps not stored");
  }

  // TODO(b/561978611): Remove this behavior. processErrors should not need a BuildViewTestCase-only
  // code path.
  @Test
  public void buildViewTest_invalidErrorKeyType_emitsAnErrorEventAndSkipsTheError()
      throws Exception {
    // Pins the BuildViewTestCase-only early skip: with a null EventBus and a key that is neither a
    // ConfiguredTargetKey nor an AspectBaseKey, the ErrorInfo is dumped to the event handler and
    // the error is otherwise ignored - processIndividualError is not called and nothing is thrown,
    // not even with --nokeep_going. Note that validation runs *before* the skip, so the exception
    // still has to be a valid analysis exception.
    ConfiguredTargetKey ctKey = configuredTargetKey("//build_info");
    ActionLookupData key = ActionLookupData.create(ctKey, /* actionIndex= */ 0);
    ErrorInfo errorInfo = errorInfo(analysisException("analysis exception", ctKey.getLabel()));

    ErrorProcessingResult result =
        processErrorsInBuildViewTest(
            resultOf(key, errorInfo), /* keepGoing= */ false, /* includeExecutionPhase= */ false);

    assertThat(errorMessages()).containsExactly(errorInfo.toString());
    // An empty result is what proves that processIndividualError was skipped: the same error on a
    // ConfiguredTargetKey would have set hasAnalysisError.
    assertThat(result.hasAnalysisError()).isFalse();
    assertThat(result.hasLoadingError()).isFalse();
    assertThat(result.executionDetailedExitCode()).isNull();
    assertThat(result.actionConflicts()).isEmpty();
    assertThat(result.aspectKeysForConflictReporting()).isEmpty();
  }

  // TODO(b/561978611): Remove this behavior. An error key this class does not recognize should not
  // be a crash.
  @Test
  public void analysisErrorOnAnInvalidErrorKeyType_throwsIllegalStateException() {
    // The other half of the skip above: with a real EventBus there is no skip, so a key that is
    // neither a ConfiguredTargetKey nor an AspectBaseKey falls through to the checkState at the
    // bottom of processIndividualError. The exception still has to be a valid analysis exception,
    // because validation runs first.
    Label label = Label.parseCanonicalUnchecked("//pkg:bad_key");
    EvaluationResult<SkyValue> result =
        resultOf(
            TransitiveTargetKey.of(label),
            errorInfo(analysisException("analysis exception", label)));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> processErrors(result, /* keepGoing= */ true, /* includeExecutionPhase= */ false));

    assertThat(thrown).hasMessageThat().contains("to be a ConfiguredTargetKey");
  }

  // TODO(b/561978611): Remove this behavior. An error key this class does not recognize should not
  // be a crash.
  @Test
  public void executionErrorOnAnInvalidErrorKeyType_throwsIllegalStateException() {
    // Same crash for an execution exception: the key type is inspected before anything notices that
    // this error has nothing to do with analysis in the first place.
    Label label = Label.parseCanonicalUnchecked("//pkg:bad_key");
    EvaluationResult<SkyValue> result =
        resultOf(
            TransitiveTargetKey.of(label),
            errorInfo(
                actionExecutionException(
                    "action failed",
                    executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE))));

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> processErrors(result, /* keepGoing= */ true, /* includeExecutionPhase= */ true));

    assertThat(thrown).hasMessageThat().contains("to be a ConfiguredTargetKey");
  }

  @Test
  public void processAnalysisErrors_keepGoing_returnsTheErrorProcessingResult() throws Exception {
    // Pins the normal return of the analysis entry point: it just forwards to processErrors with
    // includeExecutionPhase = false.
    ConfiguredTargetKey key = configuredTargetKey("//analysis_err");

    ErrorProcessingResult result =
        SkyframeErrorProcessor.processAnalysisErrors(
            resultOf(key, errorInfo(analysisException("analysis exception", key.getLabel()))),
            cyclesReporter,
            eventHandler,
            /* keepGoing= */ true,
            /* keepEdges= */ false,
            eventBus,
            bugReporter);

    assertThat(result.hasAnalysisError()).isTrue();
    assertThat(result.executionDetailedExitCode()).isNull();
  }

  @Test
  public void processAnalysisErrors_noKeepGoing_propagatesViewCreationFailedException() {
    // Pins that the entry point's catch block does not interfere with the exception it is meant to
    // let through.
    ConfiguredTargetKey key = configuredTargetKey("//pkg:analysis_err");
    ConfiguredValueCreationException cause =
        analysisException("analysis exception", key.getLabel());

    EvaluationResult<SkyValue> result = resultOf(key, errorInfo(cause));

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                SkyframeErrorProcessor.processAnalysisErrors(
                    result,
                    cyclesReporter,
                    eventHandler,
                    /* keepGoing= */ false,
                    /* keepEdges= */ false,
                    eventBus,
                    bugReporter));

    assertThat(thrown)
        .hasMessageThat()
        .contains("Analysis of target '//pkg:analysis_err' failed; build aborted");
    assertThat(thrown).hasCauseThat().isEqualTo(cause);
  }

  @Test
  public void processExecutionErrors_keepGoing_returnsTheErrorProcessingResult() throws Exception {
    // Pins the normal return of the execution entry point: it just forwards to processErrors with
    // includeExecutionPhase = true.
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    ErrorProcessingResult result =
        SkyframeErrorProcessor.processExecutionErrors(
            resultOf(key, errorInfo(actionExecutionException("action failed", exitCode))),
            cyclesReporter,
            eventHandler,
            /* keepGoing= */ true,
            /* keepEdges= */ false,
            eventBus,
            bugReporter);

    assertThat(result.executionDetailedExitCode()).isEqualTo(exitCode);
    assertThat(result.hasAnalysisError()).isFalse();
  }

  @Test
  public void processExecutionErrors_noKeepGoing_propagatesBuildFailedException() {
    ConfiguredTargetKey key = configuredTargetKey("//exec_err");
    DetailedExitCode exitCode =
        executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE);

    EvaluationResult<SkyValue> result =
        resultOf(key, errorInfo(actionExecutionExceptionWithAction("action failed", exitCode)));

    BuildFailedException thrown =
        assertThrows(
            BuildFailedException.class,
            () ->
                SkyframeErrorProcessor.processExecutionErrors(
                    result,
                    cyclesReporter,
                    eventHandler,
                    /* keepGoing= */ false,
                    /* keepEdges= */ false,
                    eventBus,
                    bugReporter));

    assertThat(thrown.getDetailedExitCode()).isEqualTo(exitCode);
  }

  @Test
  public void processExecutionErrors_noKeepGoing_propagatesTestExecException() {
    // More than coverage: this pins that the entry point's narrow catch
    // (ViewCreationFailedException
    // only) does not swallow a TestExecException on its way out.
    ConfiguredTargetKey key = configuredTargetKey("//test_err");
    TestExecException testExecException =
        new TestExecException("test failed", FailureDetail.getDefaultInstance());
    ActionExecutionException cause =
        new ActionExecutionException(
            "action failed",
            testExecException,
            /* action= */ (ActionAnalysisMetadata) null,
            /* catastrophe= */ false,
            executionExitCode("action failed", Execution.Code.ACTION_NOT_UP_TO_DATE));

    EvaluationResult<SkyValue> result = resultOf(key, errorInfo(cause));

    TestExecException thrown =
        assertThrows(
            TestExecException.class,
            () ->
                SkyframeErrorProcessor.processExecutionErrors(
                    result,
                    cyclesReporter,
                    eventHandler,
                    /* keepGoing= */ false,
                    /* keepEdges= */ false,
                    eventBus,
                    bugReporter));

    assertThat(thrown).isSameInstanceAs(testExecException);
  }

  /**
   * Same as {@link #processErrors}, but with {@code keepEdges = true}. Only useful for results that
   * carry a {@link WalkableGraph}: the "unexpected exception" path walks it.
   */
  @CanIgnoreReturnValue
  private ErrorProcessingResult processErrorsKeepingEdges(
      EvaluationResult<SkyValue> result, boolean keepGoing, boolean includeExecutionPhase)
      throws InterruptedException,
          ViewCreationFailedException,
          BuildFailedException,
          TestExecException {
    return SkyframeErrorProcessor.processErrors(
        result,
        cyclesReporter,
        eventHandler,
        keepGoing,
        /* keepEdges= */ true,
        eventBus,
        bugReporter,
        includeExecutionPhase);
  }

  // -------------------------------------------------------------------------------------------
  // Helpers.
  // -------------------------------------------------------------------------------------------

  /**
   * Runs {@link SkyframeErrorProcessor#processErrors} with a real {@link EventBus}.
   *
   * <p>{@code keepEdges} is false so that the "unexpected exception" path does not try to walk a
   * {@link com.google.devtools.build.skyframe.WalkableGraph} that these results don't have.
   */
  @CanIgnoreReturnValue
  private ErrorProcessingResult processErrors(
      EvaluationResult<SkyValue> result, boolean keepGoing, boolean includeExecutionPhase)
      throws InterruptedException,
          ViewCreationFailedException,
          BuildFailedException,
          TestExecException {
    return SkyframeErrorProcessor.processErrors(
        result,
        cyclesReporter,
        eventHandler,
        keepGoing,
        /* keepEdges= */ false,
        eventBus,
        bugReporter,
        includeExecutionPhase);
  }

  /** Same, but with a null {@link EventBus}, i.e. the {@code BuildViewTestCase} mode. */
  @CanIgnoreReturnValue
  private ErrorProcessingResult processErrorsInBuildViewTest(
      EvaluationResult<SkyValue> result, boolean keepGoing, boolean includeExecutionPhase)
      throws InterruptedException,
          ViewCreationFailedException,
          BuildFailedException,
          TestExecException {
    return SkyframeErrorProcessor.processErrors(
        result,
        cyclesReporter,
        eventHandler,
        keepGoing,
        /* keepEdges= */ false,
        /* eventBus= */ null,
        bugReporter,
        includeExecutionPhase);
  }

  private static EvaluationResult<SkyValue> resultOf(SkyKey key, ErrorInfo errorInfo) {
    return EvaluationResult.<SkyValue>builder().addError(key, errorInfo).build();
  }

  private static ErrorInfo errorInfo(Exception cause) {
    return ErrorInfo.fromException(
        new ReifiedSkyFunctionException(
            new DummySkyFunctionException(cause, Transience.PERSISTENT)),
        /* isTransitivelyTransient= */ false);
  }

  /**
   * An {@link ErrorInfo} that carries an exception <em>and</em> a cycle.
   *
   * <p>Production reaches this shape through {@link ErrorInfo#fromChildErrors}: when several
   * children of {@code key} fail - one by throwing, another by being part of a cycle - the
   * aggregated ErrorInfo keeps the first child's exception as its representative exception and
   * collects the cycles of all of them. The 5-argument {@link ErrorInfo} constructor is public and
   * would build the same thing, but going through {@code fromChildErrors} is the faithful
   * reproduction.
   */
  private static ErrorInfo errorInfo(SkyKey key, Exception cause, CycleInfo cycle) {
    return ErrorInfo.fromChildErrors(
        key, ImmutableList.of(errorInfo(cause), ErrorInfo.fromCycle(cycle)));
  }

  private static ConfiguredTargetKey configuredTargetKey(String label) {
    return ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked(label)).build();
  }

  private static AspectKey aspectKey(ConfiguredTargetKey baseKey) {
    return AspectKeyCreator.createAspectKey(
        AspectDescriptor.of(ASPECT_CLASS, AspectParameters.EMPTY), baseKey);
  }

  private static TopLevelAspectsKey topLevelAspectsKey(Label label) {
    return AspectKeyCreator.createTopLevelAspectsKey(
        ImmutableList.of(ASPECT_CLASS),
        label,
        /* configuration= */ null,
        /* topLevelAspectsParameters= */ ImmutableMap.of());
  }

  private static SkyKey wrapKey(TopLevelKeyKind kind, ConfiguredTargetKey ctKey) {
    return switch (kind) {
      case BARE_CONFIGURED_TARGET -> ctKey;
      case BUILD_DRIVER ->
          BuildDriverKey.ofConfiguredTarget(
              ctKey,
              TOP_LEVEL_ARTIFACT_CONTEXT,
              /* explicitlyRequested= */ true,
              /* skipIncompatibleExplicitTargets= */ false,
              /* extraActionTopLevelOnly= */ false,
              /* keepGoing= */ true);
      case TARGET_COMPLETION ->
          TargetCompletionValue.key(ctKey, TOP_LEVEL_ARTIFACT_CONTEXT, /* willTest= */ false);
      case TEST_COMPLETION ->
          TestCompletionValue.key(ctKey, TOP_LEVEL_ARTIFACT_CONTEXT, /* exclusiveTesting= */ false);
      case ASPECT_COMPLETION ->
          AspectCompletionKey.create(aspectKey(ctKey), TOP_LEVEL_ARTIFACT_CONTEXT);
    };
  }

  /** A cycle whose members are all actions, i.e. an execution cycle. */
  private static CycleInfo executionCycle(ConfiguredTargetKey ctKey) {
    return CycleInfo.createCycleInfo(
        ImmutableList.of(ActionLookupData.create(ctKey, /* actionIndex= */ 0)));
  }

  private static ConfiguredValueCreationException analysisException(String message, Label label) {
    return analysisExceptionWithCauses(message, label, /* rootCauses= */ null);
  }

  private static ConfiguredValueCreationException analysisExceptionWithCauses(
      String message, Label label, @Nullable NestedSet<Cause> rootCauses) {
    return new ConfiguredValueCreationException(
        /* location= */ null,
        message,
        label,
        /* configuration= */ null,
        rootCauses,
        /* detailedExitCode= */ null);
  }

  private static ActionExecutionException actionExecutionException(
      String message, DetailedExitCode detailedExitCode) {
    return new ActionExecutionException(
        message,
        /* action= */ (ActionAnalysisMetadata) null,
        /* catastrophe= */ false,
        detailedExitCode);
  }

  /**
   * Same, but with a non-null action. {@code --nokeep_going} needs one: {@link
   * SkyframeErrorProcessor#rethrow} unconditionally calls {@link
   * ActionExecutionException#getLocation}, which dereferences the action.
   *
   * <p>The action's owner has a null label, which keeps the exception's root causes empty (an owner
   * with a label would make the constructor dereference the action's primary output too).
   */
  private static ActionExecutionException actionExecutionExceptionWithAction(
      String message, DetailedExitCode detailedExitCode) {
    return actionExecutionExceptionWithAction(
        message, detailedExitCode, /* catastrophe= */ false, /* location= */ null);
  }

  /**
   * Same as {@link #actionExecutionExceptionWithAction(String, DetailedExitCode)}, but with control
   * over the message, the catastrophe bit and the owner's location.
   *
   * <p>The owner's label stays null on purpose, see the two-argument overload.
   */
  private static ActionExecutionException actionExecutionExceptionWithAction(
      @Nullable String message,
      DetailedExitCode detailedExitCode,
      boolean catastrophe,
      @Nullable Location location) {
    ActionAnalysisMetadata action = mock(ActionAnalysisMetadata.class);
    when(action.describe()).thenReturn("TestAction");
    ActionOwner owner = mock(ActionOwner.class);
    when(owner.getLocation()).thenReturn(location);
    when(action.getOwner()).thenReturn(owner);
    return new ActionExecutionException(message, action, catastrophe, detailedExitCode);
  }

  /**
   * An {@link ArtifactNestedSetEvalException}, the one {@link SkyframeErrorProcessor}
   * execution-exception type that no other test here constructs.
   *
   * <p>The nested exception set is left empty: the processor never reads it. Only the message and -
   * in {@code --nokeep_going} - the catastrophe flag are observable.
   */
  private static ArtifactNestedSetEvalException artifactNestedSetEvalException(
      String message, boolean catastrophic) {
    return new ArtifactNestedSetEvalException(
        message,
        NestedSetBuilder.<Pair<SkyKey, Exception>>emptySet(Order.STABLE_ORDER),
        catastrophic);
  }

  private static ActionConflictException actionConflictException(String message) {
    return actionConflictException(message, mock(ActionAnalysisMetadata.class));
  }

  private static ActionConflictException actionConflictException(
      String message, ActionAnalysisMetadata attemptedAction) {
    return ActionConflictException.create(
        /* artifact= */ (Artifact) null, attemptedAction, message, /* isPrefixConflict= */ false);
  }

  private static FailureDetail analysisFailureDetail(String message) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setAnalysis(Analysis.newBuilder().setCode(Analysis.Code.ANALYSIS_UNKNOWN))
        .build();
  }

  private static DetailedExitCode analysisExitCode(String message) {
    return DetailedExitCode.of(analysisFailureDetail(message));
  }

  private static DetailedExitCode executionExitCode(String message, Execution.Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(code))
            .build());
  }

  private AnalysisFailureEvent onlyAnalysisFailureEvent() {
    assertThat(eventBusCollector.analysisFailures).hasSize(1);
    return eventBusCollector.analysisFailures.get(0);
  }

  private static ImmutableList<DetailedExitCode> rootCauseDetailedExitCodes(
      AnalysisFailureEvent event) {
    return event.getRootCauses().toList().stream()
        .map(Cause::getDetailedExitCode)
        .collect(toImmutableList());
  }

  private ImmutableList<String> messagesOfKind(EventKind kind) {
    return eventHandler.getEvents().stream()
        .filter(event -> event.getKind() == kind)
        .map(Event::getMessage)
        .collect(toImmutableList());
  }

  private ImmutableList<String> warningMessages() {
    return messagesOfKind(EventKind.WARNING);
  }

  private ImmutableList<String> errorMessages() {
    return messagesOfKind(EventKind.ERROR);
  }

  /** Records the calls to {@link CyclesReporter#reportCycles}. */
  private static final class RecordingCyclesReporter extends CyclesReporter {
    private final List<SkyKey> topLevelKeys = new ArrayList<>();
    private final List<CycleInfo> cycles = new ArrayList<>();

    @Override
    public void reportCycles(
        Iterable<CycleInfo> cycleInfos, SkyKey topLevelKey, ExtendedEventHandler eventHandler) {
      topLevelKeys.add(topLevelKey);
      for (CycleInfo cycleInfo : cycleInfos) {
        cycles.add(cycleInfo);
      }
    }
  }

  /** A {@link BugReporter} that records instead of crashing the test. */
  private static final class RecordingBugReporter implements BugReporter {
    private final List<Throwable> nonFatalBugReports = new ArrayList<>();
    private final List<Throwable> bugReports = new ArrayList<>();

    @Override
    public void sendBugReport(Throwable exception, List<String> args, String... values) {
      bugReports.add(exception);
    }

    @Override
    public void sendNonFatalBugReport(Throwable exception) {
      nonFatalBugReports.add(exception);
    }

    @Override
    public void handleCrash(Crash crash, CrashContext ctx) {
      bugReports.add(crash.getThrowable());
    }
  }

  /** Records everything posted to the {@link EventBus}. */
  private static final class EventBusCollector {
    private final List<Object> allEvents = new ArrayList<>();
    private final List<AnalysisFailureEvent> analysisFailures = new ArrayList<>();
    private final List<LoadingFailureEvent> loadingFailures = new ArrayList<>();

    @Subscribe
    public void onAnyEvent(Object event) {
      allEvents.add(event);
    }

    @Subscribe
    public void onAnalysisFailure(AnalysisFailureEvent event) {
      analysisFailures.add(event);
    }

    @Subscribe
    public void onLoadingFailure(LoadingFailureEvent event) {
      loadingFailures.add(event);
    }
  }

  private static final class DummySkyFunctionException extends SkyFunctionException {
    DummySkyFunctionException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }

  /**
   * An exception type that {@link SkyframeErrorProcessor} does not know about. It has to be a
   * checked exception because {@link SkyFunctionException} rejects {@link RuntimeException} causes.
   */
  private static final class UnrecognizedException extends Exception {
    UnrecognizedException(String message) {
      super(message);
    }
  }
}
