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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertAbout;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableMap;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import java.util.Map;

/** A Truth {@link Subject} for {@link StarlarkDefinedAspect}. */
public class StarlarkDefinedAspectSubject extends Subject {
  // Static data.

  /** Entry point for test assertions related to {@link StarlarkDefinedAspect}. */
  public static StarlarkDefinedAspectSubject assertThat(
      StarlarkDefinedAspect starlarkDefinedAspect) {
    return assertAbout(StarlarkDefinedAspectSubject::new).that(starlarkDefinedAspect);
  }

  private final Map<Label, ToolchainTypeRequirement> toolchainTypesMap;

  protected StarlarkDefinedAspectSubject(
      FailureMetadata failureMetadata, StarlarkDefinedAspect subject) {
    super(failureMetadata, subject);
    this.toolchainTypesMap = makeToolchainTypesMap(subject);
  }

  private static ImmutableMap<Label, ToolchainTypeRequirement> makeToolchainTypesMap(
      StarlarkDefinedAspect subject) {
    return subject.getToolchainTypes().stream()
        .collect(toImmutableMap(ToolchainTypeRequirement::toolchainType, Functions.identity()));
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

  // TODO(blaze-team): Add more useful methods.
}
