// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.ByteString;
import java.util.Comparator;
import java.util.Map;

/** Canonical logical material embedded in the synthetic producer-keyed test action. */
public final class ProducerKeyedTestIdentity {
  public record LogicalInput(String path, String type, String identity) {}

  public record DeclaredOutput(String path, String type) {}

  private ProducerKeyedTestIdentity() {}

  public static ByteString compute(
      String producerDigest,
      long producerDigestSize,
      String testActionKey,
      String ownerLabel,
      String configurationChecksum,
      String executionPlatform,
      ImmutableMap<String, String> resolvedClientEnvironment,
      ImmutableList<DeclaredOutput> declaredOutputs,
      ImmutableList<LogicalInput> logicalInputs) {
    Fingerprint fingerprint = new Fingerprint();
    fingerprint.addString("bazel.producer_keyed_test_cache.logical_identity.v1");
    fingerprint.addString(producerDigest);
    fingerprint.addLong(producerDigestSize);
    fingerprint.addString(testActionKey);
    fingerprint.addString(ownerLabel);
    fingerprint.addString(configurationChecksum);
    fingerprint.addString(executionPlatform);
    fingerprint.addString("resolved_client_environment");
    fingerprint.addInt(resolvedClientEnvironment.size());
    resolvedClientEnvironment.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .forEach(
            entry -> {
              fingerprint.addString(entry.getKey());
              fingerprint.addString(entry.getValue());
            });
    fingerprint.addString("declared_outputs");
    fingerprint.addInt(declaredOutputs.size());
    declaredOutputs.stream()
        .sorted(Comparator.comparing(DeclaredOutput::path).thenComparing(DeclaredOutput::type))
        .forEach(
            output -> {
              fingerprint.addString(output.path());
              fingerprint.addString(output.type());
            });
    fingerprint.addString("logical_inputs");
    fingerprint.addInt(logicalInputs.size());
    logicalInputs.stream()
        .sorted(
            Comparator.comparing(LogicalInput::path)
                .thenComparing(LogicalInput::type)
                .thenComparing(LogicalInput::identity))
        .forEach(
            input -> {
              fingerprint.addString(input.path());
              fingerprint.addString(input.type());
              fingerprint.addString(input.identity());
            });
    return ByteString.copyFrom(fingerprint.digestAndReset());
  }
}
