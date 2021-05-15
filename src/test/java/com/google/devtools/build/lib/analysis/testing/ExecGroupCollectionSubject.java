// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.testing;

import static com.google.common.truth.Truth.assertAbout;
import static com.google.devtools.build.lib.analysis.testing.ExecGroupSubject.execGroups;

import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.ExecGroupCollection;

/** A Truth {@link Subject} for {@link ExecGroupCollection}. */
public class ExecGroupCollectionSubject extends Subject {
  // Static data.

  /** Entry point for test assertions related to {@link ExecGroupCollection}. */
  public static ExecGroupCollectionSubject assertThat(ExecGroupCollection execGroupCollection) {
    return assertAbout(ExecGroupCollectionSubject::new).that(execGroupCollection);
  }

  /** Static method for getting the subject factory (for use with assertAbout()). */
  public static Factory<ExecGroupCollectionSubject, ExecGroupCollection> execGroupCollection() {
    return ExecGroupCollectionSubject::new;
  }

  // Instance fields.

  private final ExecGroupCollection actual;

  protected ExecGroupCollectionSubject(
      FailureMetadata failureMetadata, ExecGroupCollection subject) {
    super(failureMetadata, subject);
    this.actual = subject;
  }

  public ExecGroupSubject execGroup(String execGroupName) {
    return check("execGroup(%s)", execGroupName)
        .about(execGroups())
        .that(actual.getExecGroup(execGroupName));
  }

  public void hasExecGroup(String execGroupName) {
    this.execGroup(execGroupName).isNotNull();
  }
}
