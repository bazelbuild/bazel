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
package com.google.devtools.build.lib.analysis.testing;

import static com.google.common.truth.Truth.assertAbout;
import static com.google.devtools.build.lib.analysis.testing.ToolchainInfoSubject.toolchainInfos;

import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.ResolvedToolchainContext;
import com.google.devtools.build.lib.cmdline.Label;

/** A Truth {@link Subject} for {@link ResolvedToolchainContext}. */
public class ResolvedToolchainContextSubject extends ToolchainContextSubject {
  // Static data.

  /** Entry point for test assertions related to {@link ResolvedToolchainContext}. */
  public static ResolvedToolchainContextSubject assertThat(
      ResolvedToolchainContext resolvedToolchainContext) {
    return assertAbout(RESOLVED_TOOLCHAIN_CONTEXT_SUBJECT_FACTORY).that(resolvedToolchainContext);
  }

  static final Factory<ResolvedToolchainContextSubject, ResolvedToolchainContext>
      RESOLVED_TOOLCHAIN_CONTEXT_SUBJECT_FACTORY = ResolvedToolchainContextSubject::new;

  // Instance fields.

  private final ResolvedToolchainContext actual;

  private ResolvedToolchainContextSubject(
      FailureMetadata failureMetadata, ResolvedToolchainContext subject) {
    super(failureMetadata, subject);
    this.actual = subject;
  }

  public ToolchainInfoSubject forToolchainType(Label toolchainType) {
    return check("forToolchainType(%s)", toolchainType)
        .about(toolchainInfos())
        .that(actual.forToolchainType(toolchainType));
  }
}
