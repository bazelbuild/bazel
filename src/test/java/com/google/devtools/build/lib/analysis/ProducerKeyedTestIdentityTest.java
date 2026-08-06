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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.test.ProducerKeyedTestIdentity;
import com.google.devtools.build.lib.analysis.test.ProducerKeyedTestIdentity.DeclaredOutput;
import com.google.devtools.build.lib.analysis.test.ProducerKeyedTestIdentity.LogicalInput;
import com.google.protobuf.ByteString;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ProducerKeyedTestIdentityTest {
  @Test
  public void identicalSemanticsProduceSameIdentity() {
    assertThat(identity("producer", "action", "owner", "config", "platform", "value", "digest"))
        .isEqualTo(
            identity("producer", "action", "owner", "config", "platform", "value", "digest"));
  }

  @Test
  public void semanticMutationsChangeIdentity() {
    ByteString baseline =
        identity("producer", "action", "owner", "config", "platform", "value", "digest");

    assertThat(identity("changed", "action", "owner", "config", "platform", "value", "digest"))
        .isNotEqualTo(baseline);
    assertThat(identity("producer", "changed", "owner", "config", "platform", "value", "digest"))
        .isNotEqualTo(baseline);
    assertThat(identity("producer", "action", "changed", "config", "platform", "value", "digest"))
        .isNotEqualTo(baseline);
    assertThat(identity("producer", "action", "owner", "changed", "platform", "value", "digest"))
        .isNotEqualTo(baseline);
    assertThat(identity("producer", "action", "owner", "config", "changed", "value", "digest"))
        .isNotEqualTo(baseline);
    assertThat(identity("producer", "action", "owner", "config", "platform", "changed", "digest"))
        .isNotEqualTo(baseline);
    assertThat(identity("producer", "action", "owner", "config", "platform", "value", "changed"))
        .isNotEqualTo(baseline);
  }

  @Test
  public void collectionOrderDoesNotChangeIdentity() {
    ByteString first =
        ProducerKeyedTestIdentity.compute(
            "producer",
            1,
            "action",
            "owner",
            "config",
            "platform",
            ImmutableMap.of("A", "a", "B", "b"),
            ImmutableList.of(new DeclaredOutput("z", "file"), new DeclaredOutput("a", "file")),
            ImmutableList.of(
                new LogicalInput("z", "source", "2"), new LogicalInput("a", "source", "1")));
    ByteString second =
        ProducerKeyedTestIdentity.compute(
            "producer",
            1,
            "action",
            "owner",
            "config",
            "platform",
            ImmutableMap.of("B", "b", "A", "a"),
            ImmutableList.of(new DeclaredOutput("a", "file"), new DeclaredOutput("z", "file")),
            ImmutableList.of(
                new LogicalInput("a", "source", "1"), new LogicalInput("z", "source", "2")));

    assertThat(first).isEqualTo(second);
  }

  @Test
  public void collectionBoundariesAreDomainSeparated() {
    ByteString environmentEntry =
        ProducerKeyedTestIdentity.compute(
            "producer",
            1,
            "action",
            "owner",
            "config",
            "platform",
            ImmutableMap.of("same", "strings"),
            ImmutableList.of(),
            ImmutableList.of());
    ByteString outputEntry =
        ProducerKeyedTestIdentity.compute(
            "producer",
            1,
            "action",
            "owner",
            "config",
            "platform",
            ImmutableMap.of(),
            ImmutableList.of(new DeclaredOutput("same", "strings")),
            ImmutableList.of());

    assertThat(environmentEntry).isNotEqualTo(outputEntry);
  }

  private static ByteString identity(
      String producer,
      String action,
      String owner,
      String config,
      String platform,
      String environmentValue,
      String inputDigest) {
    return ProducerKeyedTestIdentity.compute(
        producer,
        1,
        action,
        owner,
        config,
        platform,
        ImmutableMap.of("ENV", environmentValue),
        ImmutableList.of(new DeclaredOutput("test.log", "file")),
        ImmutableList.of(new LogicalInput("data.txt", "source", inputDigest)));
  }
}
