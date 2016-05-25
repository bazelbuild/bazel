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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.ErrorInfoSubjectFactory.assertThatErrorInfo;

import com.google.common.collect.ImmutableList;
import com.google.common.truth.DefaultSubject;
import com.google.common.truth.FailureStrategy;
import com.google.common.truth.IterableSubject;
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

  public DefaultSubject hasEntryThat(SkyKey key) {
    return assertThat(getSubject().get(key)).named("Entry for " + getDisplaySubject());
  }

  public ErrorInfoSubject hasErrorEntryForKeyThat(SkyKey key) {
    return assertThatErrorInfo(getSubject().getError(key))
        .named("Error entry for " + getDisplaySubject());
  }

  public IterableSubject hasDirectDepsInGraphThat(SkyKey parent) {
    return assertThat(
            getSubject().getWalkableGraph().getDirectDeps(ImmutableList.of(parent)).get(parent))
        .named("Direct deps for " + parent + " in " + getDisplaySubject());
  }

  public IterableSubject hasReverseDepsInGraphThat(SkyKey child) {
    return assertThat(
            getSubject().getWalkableGraph().getReverseDeps(ImmutableList.of(child)).get(child))
        .named("Reverse deps for " + child + " in " + getDisplaySubject());
  }
}
