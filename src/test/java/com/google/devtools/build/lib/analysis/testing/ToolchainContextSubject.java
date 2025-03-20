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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertAbout;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableMap;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.MapSubject;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;

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
  private final ImmutableMap<Label, ToolchainTypeRequirement> toolchainTypesMap;

  protected ToolchainContextSubject(FailureMetadata failureMetadata, ToolchainContext subject) {
    super(failureMetadata, subject);
    this.actual = subject;
    this.toolchainTypesMap = makeToolchainTypesMap(subject);
  }

  private static ImmutableMap<Label, ToolchainTypeRequirement> makeToolchainTypesMap(
      ToolchainContext subject) {
    return subject.toolchainTypes().stream()
        .collect(toImmutableMap(ToolchainTypeRequirement::toolchainType, Functions.identity()));
  }

  public void hasExecutionPlatform(String platformLabel) throws LabelSyntaxException {
    hasExecutionPlatform(Label.parseCanonical(platformLabel));
  }

  public void hasExecutionPlatform(Label platform) {
    check("executionPlatform()").that(actual.executionPlatform()).isNotNull();
    check("executionPlatform()").that(actual.executionPlatform().label()).isEqualTo(platform);
  }

  public void hasTargetPlatform(String platformLabel) throws LabelSyntaxException {
    hasTargetPlatform(Label.parseCanonical(platformLabel));
  }

  public void hasTargetPlatform(Label platform) {
    check("targetPlatform()").that(actual.targetPlatform()).isNotNull();
    check("targetPlatform()").that(actual.targetPlatform().label()).isEqualTo(platform);
  }

  public MapSubject toolchainTypes() {
    return check("toolchainTypes()").that(toolchainTypesMap);
  }

  public ToolchainTypeRequirementSubject toolchainType(String toolchainTypeLabel) {
    return toolchainType(Label.parseCanonicalUnchecked(toolchainTypeLabel));
  }

  public ToolchainTypeRequirementSubject toolchainType(Label toolchainType) {
    return check("toolchainType(%s)", toolchainType)
        .about(ToolchainTypeRequirementSubject.toolchainTypeRequirements())
        .that(toolchainTypesMap.get(toolchainType));
  }

  public void hasToolchainType(String toolchainTypeLabel) {
    toolchainType(toolchainTypeLabel).isNotNull();
  }

  public void hasToolchainType(Label toolchainType) {
    toolchainType(toolchainType).isNotNull();
  }

  public void doesntHaveToolchainType(String toolchainTypeLabel) {
    doesntHaveToolchainType(Label.parseCanonicalUnchecked(toolchainTypeLabel));
  }

  public void doesntHaveToolchainType(Label toolchainType) {
    check("toolchainType(%s)", toolchainType)
        .that(toolchainTypesMap.containsKey(toolchainType))
        .isFalse();
  }

  public void hasResolvedToolchain(String resolvedToolchainLabel) throws LabelSyntaxException {
    hasResolvedToolchain(Label.parseCanonical(resolvedToolchainLabel));
  }

  public void hasResolvedToolchain(Label resolvedToolchain) {
    resolvedToolchainLabels().contains(resolvedToolchain);
  }

  public IterableSubject resolvedToolchainLabels() {
    return check("resolvedToolchainLabels()").that(actual.resolvedToolchainLabels());
  }
}
