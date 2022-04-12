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

import com.google.common.collect.ImmutableMap;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import java.util.stream.Collectors;

/** A Truth {@link Subject} for {@link ToolchainContext}. */
public class ToolchainContextSubject extends Subject {
  // Static data.

  /** Entry point for test assertions related to {@link ToolchainContext}. */
  public static ToolchainContextSubject assertThat(ToolchainContext toolchainContext) {
    return assertAbout(ToolchainContextSubject::new).that(toolchainContext);
  }

  /** Static method for getting the subject factory (for use with assertAbout()). */
  public static Subject.Factory<ToolchainContextSubject, ToolchainContext> toolchainContexts() {
    return ToolchainContextSubject::new;
  }

  // Instance fields.

  private final ToolchainContext actual;

  protected ToolchainContextSubject(FailureMetadata failureMetadata, ToolchainContext subject) {
    super(failureMetadata, subject);
    this.actual = subject;
  }

  public void hasExecutionPlatform(String platformLabel) throws LabelSyntaxException {
    hasExecutionPlatform(Label.parseAbsolute(platformLabel, ImmutableMap.of()));
  }

  public void hasExecutionPlatform(Label platform) {
    check("executionPlatform()").that(actual.executionPlatform()).isNotNull();
    check("executionPlatform()").that(actual.executionPlatform().label()).isEqualTo(platform);
  }

  public void hasTargetPlatform(String platformLabel) throws LabelSyntaxException {
    hasTargetPlatform(Label.parseAbsolute(platformLabel, ImmutableMap.of()));
  }

  public void hasTargetPlatform(Label platform) {
    check("targetPlatform()").that(actual.targetPlatform()).isNotNull();
    check("targetPlatform()").that(actual.targetPlatform().label()).isEqualTo(platform);
  }

  public void hasToolchainType(String toolchainTypeLabel) throws LabelSyntaxException {
    hasToolchainType(Label.parseAbsolute(toolchainTypeLabel, ImmutableMap.of()));
  }

  public void hasToolchainType(Label toolchainType) {
    toolchainTypeLabels().contains(toolchainType);
  }

  public IterableSubject toolchainTypeLabels() {
    return check("requiredToolchainTypes()")
        .that(
            actual.requiredToolchainTypes().stream()
                .map(ToolchainTypeInfo::typeLabel)
                .collect(Collectors.toList()));
  }

  public void hasResolvedToolchain(String resolvedToolchainLabel) throws LabelSyntaxException {
    hasResolvedToolchain(Label.parseAbsolute(resolvedToolchainLabel, ImmutableMap.of()));
  }

  public void hasResolvedToolchain(Label resolvedToolchain) {
    resolvedToolchainLabels().contains(resolvedToolchain);
  }

  public IterableSubject resolvedToolchainLabels() {
    return check("resolevdToolchainLabels()").that(actual.resolvedToolchainLabels());
  }
}
