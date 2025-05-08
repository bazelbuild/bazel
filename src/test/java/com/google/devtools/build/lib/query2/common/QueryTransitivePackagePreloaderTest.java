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

package com.google.devtools.build.lib.query2.common;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.skyframe.TransitiveTargetKey;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.List;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentMatchers;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link QueryTransitivePackagePreloader}. */
@RunWith(TestParameterInjector.class)
public class QueryTransitivePackagePreloaderTest {
  private static final Label LABEL = Label.parseCanonicalUnchecked("//my:label");
  private static final Label LABEL2 = Label.parseCanonicalUnchecked("//my:label2");
  private static final Label LABEL3 = Label.parseCanonicalUnchecked("//my:label3");
  private static final TransitiveTargetKey KEY = TransitiveTargetKey.of(LABEL);
  private static final TransitiveTargetKey KEY2 = TransitiveTargetKey.of(LABEL2);
  private static final TransitiveTargetKey KEY3 = TransitiveTargetKey.of(LABEL3);

  private static final ErrorInfo DETAILED_ERROR =
      ErrorInfo.fromException(
          new SkyFunctionException.ReifiedSkyFunctionException(
              new SkyFunctionException(
                  new MyDetailedException("bork"), SkyFunctionException.Transience.PERSISTENT) {}),
          /*isTransitivelyTransient=*/ false);
  private static final ErrorInfo UNDETAILED_ERROR =
      ErrorInfo.fromException(
          new SkyFunctionException.ReifiedSkyFunctionException(
              new SkyFunctionException(
                  new UndetailedException("bork"), SkyFunctionException.Transience.PERSISTENT) {}),
          /*isTransitivelyTransient=*/ false);
  private static final ErrorInfo CYCLE_ERROR =
      ErrorInfo.fromCycle(new CycleInfo(ImmutableList.of(KEY)));

  @Mock MemoizingEvaluator memoizingEvaluator;
  @Mock EvaluationContext.Builder contextBuilder;
  @Mock EvaluationContext context;
  private final BugReporter bugReporter = mock(BugReporter.class);

  private final QueryTransitivePackagePreloader underTest =
      new QueryTransitivePackagePreloader(
          () -> memoizingEvaluator, () -> contextBuilder, bugReporter);
  private AutoCloseable closeable;

  @Before
  public void setUpMocks() {
    closeable = MockitoAnnotations.openMocks(this);
    when(contextBuilder.setKeepGoing(ArgumentMatchers.anyBoolean())).thenReturn(contextBuilder);
    when(contextBuilder.setParallelism(ArgumentMatchers.anyInt())).thenReturn(contextBuilder);
    when(contextBuilder.setEventHandler(ArgumentMatchers.any())).thenReturn(contextBuilder);
    when(contextBuilder.build()).thenReturn(context);
  }

  @After
  public void releaseMocks() throws Exception {
    verifyNoMoreInteractions(memoizingEvaluator);
    verifyNoMoreInteractions(bugReporter);
    closeable.close();
  }

  @Test
  public void preloadTransitiveTargets_noError() throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(EvaluationResult.builder().build());

    underTest.preloadTransitiveTargets(
        mock(ExtendedEventHandler.class),
        ImmutableList.of(LABEL),
        /*keepGoing=*/ true,
        1,
        /*callerForError=*/ null);

    verify(memoizingEvaluator).evaluate(roots, context);
  }

  @Test
  public void preloadTransitiveTargets_errorWithNullCallerKeepGoing_doesntCleanGraph()
      throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(EvaluationResult.builder().addError(KEY, UNDETAILED_ERROR).build());

    underTest.preloadTransitiveTargets(
        mock(ExtendedEventHandler.class),
        ImmutableList.of(LABEL),
        /*keepGoing=*/ true,
        1,
        /*callerForError=*/ null);

    verify(memoizingEvaluator).evaluate(roots, context);
  }

  @Test
  public void preloadTransitiveTargets_errorWithNullCallerKeepGoingCatastrophe_cleansGraph()
      throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(
            EvaluationResult.builder()
                .setCatastrophe(new UndetailedException("catas"))
                .addError(KEY, UNDETAILED_ERROR)
                .build());

    underTest.preloadTransitiveTargets(
        mock(ExtendedEventHandler.class),
        ImmutableList.of(LABEL),
        /*keepGoing=*/ true,
        1,
        /*callerForError=*/ null);

    verify(memoizingEvaluator).evaluate(roots, context);
    verify(memoizingEvaluator).evaluate(ImmutableList.of(), context);
  }

  @Test
  public void preloadTransitiveTargets_errorWithNullCallerNoKeepGoing_cleansGraph()
      throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(EvaluationResult.builder().addError(KEY, UNDETAILED_ERROR).build());

    underTest.preloadTransitiveTargets(
        mock(ExtendedEventHandler.class),
        ImmutableList.of(LABEL),
        /*keepGoing=*/ false,
        1,
        /*callerForError=*/ null);

    verify(memoizingEvaluator).evaluate(roots, context);
    verify(memoizingEvaluator).evaluate(ImmutableList.of(), context);
  }

  @Test
  public void preloadTransitiveTargets_detailedErrorWithCaller_throwsError(
      @TestParameter boolean keepGoing) throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(EvaluationResult.builder().addError(KEY, DETAILED_ERROR).build());

    var e =
        assertThrows(
            QueryException.class,
            () ->
                underTest.preloadTransitiveTargets(
                    mock(ExtendedEventHandler.class),
                    ImmutableList.of(LABEL),
                    keepGoing,
                    1,
                    /*callerForError=*/ mock(QueryExpression.class)));
    assertThat(e).hasMessageThat().contains("failed: bork");
    assertThat(e.getFailureDetail())
        .isSameInstanceAs(MyDetailedException.DETAILED_EXIT_CODE.getFailureDetail());

    verify(memoizingEvaluator).evaluate(roots, context);
  }

  @Test
  public void preloadTransitiveTargets_undetailedErrorWithCaller_throwsErrorAndFilesBugReport(
      @TestParameter boolean keepGoing) throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(EvaluationResult.builder().addError(KEY, UNDETAILED_ERROR).build());

    var e =
        assertThrows(
            QueryException.class,
            () ->
                underTest.preloadTransitiveTargets(
                    mock(ExtendedEventHandler.class),
                    ImmutableList.of(LABEL),
                    keepGoing,
                    1,
                    /*callerForError=*/ mock(QueryExpression.class)));
    assertThat(e).hasMessageThat().contains("failed: bork");
    assertThat(e.getFailureDetail())
        .comparingExpectedFieldsOnly()
        .isEqualTo(
            FailureDetails.FailureDetail.newBuilder()
                .setQuery(
                    FailureDetails.Query.newBuilder()
                        .setCode(FailureDetails.Query.Code.NON_DETAILED_ERROR)
                        .build())
                .build());

    verify(memoizingEvaluator).evaluate(roots, context);
    verify(bugReporter).sendNonFatalBugReport(ArgumentMatchers.any());
  }

  @Test
  public void
      preloadTransitiveTargets_undetailedCatastropheAndDetailedExceptionWithCaller_throwsErrorAndFilesBugReport(
          @TestParameter boolean keepGoing) throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(
            EvaluationResult.builder()
                .addError(KEY, DETAILED_ERROR)
                .setCatastrophe(new UndetailedException("undetailed bok"))
                .build());

    var e =
        assertThrows(
            QueryException.class,
            () ->
                underTest.preloadTransitiveTargets(
                    mock(ExtendedEventHandler.class),
                    ImmutableList.of(LABEL),
                    keepGoing,
                    1,
                    /*callerForError=*/ mock(QueryExpression.class)));
    assertThat(e).hasMessageThat().contains("failed: undetailed bok");
    assertThat(e.getFailureDetail())
        .comparingExpectedFieldsOnly()
        .isEqualTo(
            FailureDetails.FailureDetail.newBuilder()
                .setQuery(
                    FailureDetails.Query.newBuilder()
                        .setCode(FailureDetails.Query.Code.NON_DETAILED_ERROR)
                        .build())
                .build());

    verify(memoizingEvaluator).evaluate(roots, context);
    verify(bugReporter).sendNonFatalBugReport(ArgumentMatchers.any());
  }

  @Test
  public void preloadTransitiveTargets_undetailedAndDetailedExceptionsWithCaller_throwsError(
      @TestParameter boolean keepGoing, @TestParameter boolean includeCycle) throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY, KEY2, KEY3);

    EvaluationResult.Builder<SkyValue> resultBuilder =
        EvaluationResult.builder().addError(KEY, UNDETAILED_ERROR).addError(KEY2, DETAILED_ERROR);
    if (includeCycle) {
      resultBuilder.addError(KEY3, CYCLE_ERROR);
    }
    when(memoizingEvaluator.evaluate(roots, context)).thenReturn(resultBuilder.build());

    var e =
        assertThrows(
            QueryException.class,
            () ->
                underTest.preloadTransitiveTargets(
                    mock(ExtendedEventHandler.class),
                    ImmutableList.of(LABEL, LABEL2, LABEL3),
                    keepGoing,
                    1,
                    /*callerForError=*/ mock(QueryExpression.class)));
    assertThat(e).hasMessageThat().contains("failed: bork");
    assertThat(e.getFailureDetail())
        .isSameInstanceAs(MyDetailedException.DETAILED_EXIT_CODE.getFailureDetail());

    verify(memoizingEvaluator).evaluate(roots, context);
  }

  @Test
  public void preloadTransitiveTargets_cycleOnly_returns() throws Exception {
    List<TransitiveTargetKey> roots = Lists.newArrayList(KEY);

    when(memoizingEvaluator.evaluate(roots, context))
        .thenReturn(EvaluationResult.builder().addError(KEY, CYCLE_ERROR).build());

    underTest.preloadTransitiveTargets(
        mock(ExtendedEventHandler.class),
        ImmutableList.of(LABEL),
        /*keepGoing=*/ true,
        1,
        /*callerForError=*/ null);

    verify(memoizingEvaluator).evaluate(roots, context);
  }

  private static final class UndetailedException extends Exception {
    UndetailedException(String message) {
      super(message);
    }
  }

  private static final class MyDetailedException extends Exception implements DetailedException {
    private static final DetailedExitCode DETAILED_EXIT_CODE =
        DetailedExitCode.of(
            FailureDetails.FailureDetail.newBuilder()
                .setQuery(
                    FailureDetails.Query.newBuilder()
                        .setCode(FailureDetails.Query.Code.BUILD_FILE_ERROR))
                .build());

    MyDetailedException(String message) {
      super(message);
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return DETAILED_EXIT_CODE;
    }
  }
}
