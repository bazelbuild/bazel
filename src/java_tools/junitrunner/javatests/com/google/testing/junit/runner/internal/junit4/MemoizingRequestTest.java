// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.internal.junit4;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.RETURNS_MOCKS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.verifyZeroInteractions;

import junit.framework.TestCase;
import org.junit.runner.Request;
import org.junit.runner.Runner;

/**
 * Tests for {@code MemoizingRequest}.
 */
public class MemoizingRequestTest extends TestCase {
  private Request mockRequestDelegate;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    mockRequestDelegate = mock(Request.class, RETURNS_MOCKS);
  }

  public void testConstructorDoesNoWork() {
    new MemoizingRequest(mockRequestDelegate);

    verifyZeroInteractions(mockRequestDelegate);
  }

  public void testMemoizesRunner() {
    MemoizingRequest memoizingRequest = new MemoizingRequest(mockRequestDelegate);

    Runner firstRunner = memoizingRequest.getRunner();
    Runner secondRunner = memoizingRequest.getRunner();

    assertThat(secondRunner).isSameInstanceAs(firstRunner);
    verify(mockRequestDelegate).getRunner();
    verifyNoMoreInteractions(mockRequestDelegate);
  }

  public void testOverridingCreateRunner() {
    final Runner stubRunner = mock(Runner.class);
    MemoizingRequest memoizingRequest = new MemoizingRequest(mockRequestDelegate) {
      @Override
      protected Runner createRunner(Request delegate) {
        return stubRunner;
      }
    };

    Runner firstRunner = memoizingRequest.getRunner();
    Runner secondRunner = memoizingRequest.getRunner();

    assertThat(firstRunner).isSameInstanceAs(stubRunner);
    assertThat(secondRunner).isSameInstanceAs(firstRunner);
    verifyZeroInteractions(mockRequestDelegate);
  }
}
