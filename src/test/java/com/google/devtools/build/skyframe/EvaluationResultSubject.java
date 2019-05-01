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

import static com.google.common.truth.Fact.simpleFact;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.skyframe.ErrorInfoSubjectFactory.assertThatErrorInfo;

import com.google.common.collect.ImmutableList;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;

/**
 * {@link Subject} for {@link EvaluationResult}. Please add to this class if you need more
 * functionality!
 */
public class EvaluationResultSubject extends Subject<EvaluationResultSubject, EvaluationResult<?>> {
  public EvaluationResultSubject(
      FailureMetadata failureMetadata, EvaluationResult<?> evaluationResult) {
    super(failureMetadata, evaluationResult);
  }

  public void hasError() {
    if (!getSubject().hasError()) {
      failWithActual(simpleFact("expected to have error"));
    }
  }

  public void hasNoError() {
    if (getSubject().hasError()) {
      failWithActual(simpleFact("expected to have no error"));
    }
  }

  public Subject<?, ?> hasEntryThat(SkyKey key) {
    return assertWithMessage("Entry for " + actualAsString()).that(getSubject().get(key));
  }

  public ErrorInfoSubject hasErrorEntryForKeyThat(SkyKey key) {
    return assertThatErrorInfo(getSubject().getError(key))
        .named("Error entry for " + actualAsString());
  }

  public IterableSubject hasDirectDepsInGraphThat(SkyKey parent) throws InterruptedException {
    return assertWithMessage("Direct deps for " + parent + " in " + actualAsString())
        .that(getSubject().getWalkableGraph().getDirectDeps(ImmutableList.of(parent)).get(parent));
  }

  public IterableSubject hasReverseDepsInGraphThat(SkyKey child) throws InterruptedException {
    return assertWithMessage("Reverse deps for " + child + " in " + actualAsString())
        .that(getSubject().getWalkableGraph().getReverseDeps(ImmutableList.of(child)).get(child));
  }
}
