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

import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;
import com.google.common.truth.ThrowableSubject;

/**
 * {@link Subject} for {@link ErrorInfo}. Please add to this class if you need more
 * functionality!
 */
public class ErrorInfoSubject extends Subject<ErrorInfoSubject, ErrorInfo> {
  public ErrorInfoSubject(FailureMetadata failureMetadata, ErrorInfo errorInfo) {
    super(failureMetadata, errorInfo);
  }

  public ThrowableSubject hasExceptionThat() {
    return check("getException()")
        .withMessage("Exception in " + actualAsString())
        .that(getSubject().getException());
  }

  public IterableSubject hasCycleInfoThat() {
    isNotNull();
    return check("getCycleInfo()")
        .withMessage("CycleInfo in " + actualAsString())
        .that(getSubject().getCycleInfo());
  }

  public void rootCauseOfExceptionIs(SkyKey key) {
    check("getRootCauseOfException()").that(getSubject().getRootCauseOfException()).isEqualTo(key);
  }

  public void isTransient() {
    if (!getSubject().isTransitivelyTransient()) {
      failWithActual(simpleFact("expected to be transient"));
    }
  }

  public void isNotTransient() {
    if (getSubject().isTransitivelyTransient()) {
      failWithActual(simpleFact("expected not to be transient"));
    }
  }
}
