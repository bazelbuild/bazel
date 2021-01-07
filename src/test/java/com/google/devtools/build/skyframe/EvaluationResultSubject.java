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

import com.google.common.collect.ImmutableList;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.MapSubject;
import com.google.common.truth.Subject;

/**
 * {@link Subject} for {@link EvaluationResult}. Please add to this class if you need more
 * functionality!
 */
public class EvaluationResultSubject extends Subject {
  private final EvaluationResult<?> actual;

  public EvaluationResultSubject(
      FailureMetadata failureMetadata, EvaluationResult<?> evaluationResult) {
    super(failureMetadata, evaluationResult);
    this.actual = evaluationResult;
  }

  public void hasError() {
    if (!actual.hasError()) {
      failWithActual(simpleFact("expected to have error"));
    }
  }

  public void hasNoError() {
    if (actual.hasError()) {
      failWithActual(simpleFact("expected to have no error"));
    }
  }

  public Subject hasEntryThat(SkyKey key) {
    return check("get(%s)", key).that(actual.get(key));
  }

  public ErrorInfoSubject hasErrorEntryForKeyThat(SkyKey key) {
    return check("getError(%s)", key)
        .about(new ErrorInfoSubjectFactory())
        .that(actual.getError(key));
  }

  public IterableSubject hasDirectDepsInGraphThat(SkyKey parent) throws InterruptedException {
    return check("directDeps(%s)", parent)
        .that(actual.getWalkableGraph().getDirectDeps(ImmutableList.of(parent)).get(parent));
  }

  public IterableSubject hasReverseDepsInGraphThat(SkyKey child) throws InterruptedException {
    return check("reverseDeps(%s)", child)
        .that(actual.getWalkableGraph().getReverseDeps(ImmutableList.of(child)).get(child));
  }

  public MapSubject hasErrorMapThat() {
    return check("errorMap()").that(actual.errorMap());
  }

  public ErrorInfoSubject hasSingletonErrorThat(SkyKey key) {
    hasError();
    hasErrorMapThat().hasSize(1);
    check("keyNames()").that(actual.keyNames()).isEmpty();
    return hasErrorEntryForKeyThat(key);
  }
}
