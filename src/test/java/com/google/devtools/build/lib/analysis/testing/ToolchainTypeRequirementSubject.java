// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.truth.ComparableSubject;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;

/** A Truth {@link Subject} for {@link ToolchainTypeRequirement}. */
public class ToolchainTypeRequirementSubject extends Subject {
  // Static data.

  /** Entry point for test assertions related to {@link ToolchainTypeRequirement}. */
  public static ToolchainTypeRequirementSubject assertThat(
      ToolchainTypeRequirement toolchainTypeRequirement) {
    return assertAbout(ToolchainTypeRequirementSubject::new).that(toolchainTypeRequirement);
  }

  /** Static method for getting the subject factory (for use with assertAbout()). */
  public static Subject.Factory<ToolchainTypeRequirementSubject, ToolchainTypeRequirement>
      toolchainTypeRequirements() {
    return ToolchainTypeRequirementSubject::new;
  }

  // Instance fields.

  private final ToolchainTypeRequirement actual;

  protected ToolchainTypeRequirementSubject(
      FailureMetadata failureMetadata, ToolchainTypeRequirement subject) {
    super(failureMetadata, subject);
    this.actual = subject;
  }

  public ComparableSubject<Label> toolchainType() {
    return check("toolchainType").that(actual.toolchainType());
  }

  public void isMandatory() {
    check("mandatory").that(actual.mandatory()).isTrue();
  }

  public void isOptional() {
    check("mandatory").that(actual.mandatory()).isFalse();
  }
}
