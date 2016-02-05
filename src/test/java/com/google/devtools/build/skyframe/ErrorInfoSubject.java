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
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;
import com.google.common.truth.ThrowableSubject;
import com.google.common.truth.Truth;

/**
 * {@link Subject} for {@link ErrorInfo}. Please add to this class if you need more
 * functionality!
 */
public class ErrorInfoSubject extends Subject<ErrorInfoSubject, ErrorInfo> {
  public ErrorInfoSubject(FailureStrategy failureStrategy, ErrorInfo errorInfo) {
    super(failureStrategy, errorInfo);
  }

  public ThrowableSubject hasExceptionThat() {
    return Truth.assertThat(getSubject().getException())
        .named("Exception in " + getDisplaySubject());
  }

  public IterableSubject hasCycleInfoThat() {
    return Truth.assertThat(getSubject().getCycleInfo())
        .named("CycleInfo in " + getDisplaySubject());
  }

  public void rootCauseOfExceptionIs(SkyKey key) {
    if (!getSubject().getRootCauseOfException().equals(key)) {
      fail("has root cause of exception " + key);
    }
  }
}
