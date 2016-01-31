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
package com.google.devtools.build.skyframe;

import com.google.common.truth.FailureStrategy;
import com.google.common.truth.Subject;

/**
 * {@link Subject} for {@link EvaluationResult}. Please add to this class if you need more
 * functionality!
 */
public class EvaluationResultSubject extends Subject<EvaluationResultSubject, EvaluationResult<?>> {
  public EvaluationResultSubject(
      FailureStrategy failureStrategy, EvaluationResult<?> evaluationResult) {
    super(failureStrategy, evaluationResult);
  }

  public void hasError() {
    if (!getSubject().hasError()) {
      fail("has error");
    }
  }

  public void hasNoError() {
    if (getSubject().hasError()) {
      fail("has no error");
    }
  }
}
