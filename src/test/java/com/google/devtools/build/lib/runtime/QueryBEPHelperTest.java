// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static org.mockito.Matchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyZeroInteractions;
import static org.mockito.Mockito.when;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.query2.CommonQueryOptions;
import com.google.devtools.build.lib.query2.QueryOutputEvent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParser;
import java.io.OutputStream;
import org.hamcrest.BaseMatcher;
import org.hamcrest.Description;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Simple unit tests for {@link QueryBEPHelper}. */
@RunWith(JUnit4.class)
public class QueryBEPHelperTest {
  private CommonQueryOptions commonQueryOptions;

  @Before
  public void setUpDefaultCommonQueryOptions() {
    commonQueryOptions =
        OptionsParser.newOptionsParser(CommonQueryOptions.class)
            .getOptions(CommonQueryOptions.class);
  }

  @Test
  public void usesStdoutOutputStream() throws Exception {
    EventBus mockEventBus = mock(EventBus.class);
    commonQueryOptions.uploadQueryOutputUsingBEP = false;
    Path mockQueryOutputFilePath = mock(Path.class);
    OutputStream mockOutputStreamForOutputPath = mock(OutputStream.class);
    OutputStream mockStdoutOutputStream = mock(OutputStream.class);

    when(mockQueryOutputFilePath.getOutputStream()).thenReturn(mockOutputStreamForOutputPath);

    QueryBEPHelper underTest = QueryBEPHelper.createForUnitTests(
        "dummy_command_name",
        mockEventBus,
        commonQueryOptions,
        mockQueryOutputFilePath,
        mockStdoutOutputStream);
    assertThat(underTest.getOutputStreamForQueryOutput()).isSameAs(mockStdoutOutputStream);
    underTest.afterQueryOutputIsWritten();
    underTest.close();

    verify(mockEventBus, times(1)).post(
        argThat(new QueryOutputEventMatcher(QueryOutputEvent.forOutputWrittenToStdout())));
    verifyZeroInteractions(mockStdoutOutputStream);
    verifyZeroInteractions(mockQueryOutputFilePath);
  }

  @Test
  public void usesQueryOutputFilePathAndClosesOutputStream() throws Exception {
    EventBus mockEventBus = mock(EventBus.class);
    commonQueryOptions.uploadQueryOutputUsingBEP = true;
    Path mockQueryOutputFilePath = mock(Path.class);
    OutputStream mockOutputStreamForOutputPath = mock(OutputStream.class);
    OutputStream mockStdoutOutputStream = mock(OutputStream.class);

    when(mockQueryOutputFilePath.getOutputStream()).thenReturn(mockOutputStreamForOutputPath);

    QueryBEPHelper underTest = QueryBEPHelper.createForUnitTests(
        "dummy_command_name",
        mockEventBus,
        commonQueryOptions,
        mockQueryOutputFilePath,
        mockStdoutOutputStream);
    assertThat(underTest.getOutputStreamForQueryOutput()).isSameAs(mockOutputStreamForOutputPath);
    underTest.afterQueryOutputIsWritten();
    underTest.close();


    verify(mockEventBus, times(1))
        .post(argThat(new QueryOutputEventMatcher(QueryOutputEvent.forOutputWrittenToFile(
            "dummy_command_name", mockQueryOutputFilePath))));
    verify(mockQueryOutputFilePath, times(1)).getOutputStream();
    verify(mockOutputStreamForOutputPath, times(1)).close();
    verifyZeroInteractions(mockStdoutOutputStream);
  }

  private static class QueryOutputEventMatcher extends BaseMatcher<QueryOutputEvent> {
    private final QueryOutputEvent queryOutputEvent;

    private QueryOutputEventMatcher(QueryOutputEvent queryOutputEvent) {
      this.queryOutputEvent = queryOutputEvent;
    }

    @Override
    public boolean matches(Object item) {
      if (!(item instanceof QueryOutputEvent)) {
        return false;
      }
      return queryOutputEvent.equalsForTesting((QueryOutputEvent) item);
    }

    @Override
    public void describeTo(Description description) {
      description.appendText(String.format("Wanted %s", queryOutputEvent));
    }
  }
}
