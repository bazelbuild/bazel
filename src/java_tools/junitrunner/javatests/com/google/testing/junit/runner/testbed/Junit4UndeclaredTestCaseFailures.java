// Copyright 2023 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.testbed;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;

/** A JUnit4-style test meant to be invoked by junit4_testbridge_tests.sh. */
@RunWith(MockitoJUnitRunner.Strict.class)
public class Junit4UndeclaredTestCaseFailures {

  @SuppressWarnings("unchecked")
  private final List<String> mockList = mock(List.class);

  @Test
  public void passesButHasUnnecessaryStubs() {
    when(mockList.add("")).thenReturn(true); // this won't get called
  }
}
