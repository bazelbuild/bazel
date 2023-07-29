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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;

import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Tests for {@link SkyframeErrorProcessor}.
 *
 * <p>TODO(b/221024798): Improve test coverage.
 */
@RunWith(TestParameterInjector.class)
public class SkyframeErrorProcessorTest {

  @Test
  public void testProcessErrors_analysisErrorNoKeepGoing_throwsException(
      @TestParameter boolean includeExecutionPhase) throws Exception {
    ConfiguredTargetKey analysisErrorKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//analysis_err"))
            .build();
    ConfiguredValueCreationException analysisException =
        new ConfiguredValueCreationException(
            new TargetAndConfiguration(mock(Target.class), /* configuration= */ null),
            "analysis exception");
    ErrorInfo analysisErrorInfo =
        ErrorInfo.fromException(
            new ReifiedSkyFunctionException(
                new DummySkyFunctionException(analysisException, Transience.PERSISTENT)),
            /*isTransitivelyTransient=*/ false);

    EvaluationResult<SkyValue> result =
        EvaluationResult.builder().addError(analysisErrorKey, analysisErrorInfo).build();

    ViewCreationFailedException thrown =
        assertThrows(
            ViewCreationFailedException.class,
            () ->
                SkyframeErrorProcessor.processErrors(
                    result,
                    /*cyclesReporter=*/ new CyclesReporter(),
                    /*eventHandler=*/ mock(ExtendedEventHandler.class),
                    /*keepGoing=*/ false,
                    /*eventBus=*/ null,
                    /*bugReporter=*/ null,
                    includeExecutionPhase));
    assertThat(thrown).hasCauseThat().isEqualTo(analysisException);
  }

  private static final class DummySkyFunctionException extends SkyFunctionException {
    DummySkyFunctionException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }
}
