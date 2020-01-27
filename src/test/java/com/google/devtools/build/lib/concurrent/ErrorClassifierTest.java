// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.concurrent.ErrorClassifier.ErrorClassification;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ErrorClassifier}. */
@RunWith(JUnit4.class)
public class ErrorClassifierTest {
  @Test
  public void testErrorClassificationNaturalOrder() {
    ErrorClassification[] values = ErrorClassification.values();
    Arrays.sort(values);
    assertThat(values).asList().containsExactly(
        ErrorClassification.NOT_CRITICAL,
        ErrorClassification.CRITICAL,
        ErrorClassification.CRITICAL_AND_LOG,
        ErrorClassification.AS_CRITICAL_AS_POSSIBLE).inOrder();
  }
}

