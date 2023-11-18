// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.util;

import com.google.common.collect.ImmutableMap;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.packages.TargetData;
import java.util.Optional;

/** Truth subject for {@link TargetData}. */
public final class TargetDataSubject extends Subject {
  private final TargetData targetData;

  private TargetDataSubject(FailureMetadata failureMetadata, TargetData targetData) {
    super(failureMetadata, targetData);
    this.targetData = targetData;
  }

  public static TargetDataSubject assertThat(TargetData targetData) {
    return Truth.assertAbout(TargetDataSubject::new).that(targetData);
  }

  public void hasSamePropertiesAs(TargetData that) {
    Truth.assertThat(toMap(targetData)).isEqualTo(toMap(that));
  }

  /** A test helper to help verify that two {@link TargetData} instances are the same. */
  private static ImmutableMap<String, Object> toMap(TargetData targetData) {
    return ImmutableMap.<String, Object>builder()
        .put("targetKind", targetData.getTargetKind())
        .put("ruleClass", targetData.getRuleClass())
        .put("label", targetData.getLabel())
        .put("isRule", targetData.isRule())
        .put("isFile", targetData.isFile())
        .put("isInputFile", targetData.isInputFile())
        .put("isOutputFile", targetData.isOutputFile())
        .put("generatingRuleLabel", Optional.ofNullable(targetData.getGeneratingRuleLabel()))
        .put("inputPath", Optional.ofNullable(targetData.getInputPath()))
        .put("deprecationWarning", Optional.ofNullable(targetData.getDeprecationWarning()))
        .put("isTestOnly", targetData.isTestOnly())
        .put("testTimeout", Optional.ofNullable(targetData.getTestTimeout()))
        .put("advertisedProviders", targetData.getAdvertisedProviders())
        .buildOrThrow();
  }
}
