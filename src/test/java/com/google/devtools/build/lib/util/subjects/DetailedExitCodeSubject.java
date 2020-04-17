// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.subjects;

import static com.google.common.truth.Fact.simpleFact;

import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;

/** A Truth-compatible {@link Subject} for {@link DetailedExitCode}. */
public class DetailedExitCodeSubject extends Subject {

  private final DetailedExitCode actual;

  public DetailedExitCodeSubject(FailureMetadata failureMetadata, DetailedExitCode exitCode) {
    super(failureMetadata, exitCode);
    this.actual = exitCode;
  }

  public void hasExitCode(ExitCode exitCode) {
    isNotNull();
    check("getExitCode()").that(actual.getExitCode()).isEqualTo(exitCode);
  }

  public void isSuccessful() {
    if (!actual.isSuccess()) {
      failWithActual(simpleFact("expected to be SUCCESS"));
    }
  }

  public void isNotSuccessful() {
    if (actual.isSuccess()) {
      failWithActual(simpleFact("expected *not* to be SUCCESS"));
    }
  }
}
