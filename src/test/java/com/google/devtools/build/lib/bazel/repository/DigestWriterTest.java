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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.bazel.bzlmod.AttributeValues;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.EnvironmentVariableValue;
import com.google.devtools.build.skyframe.AbstractSkyFunctionEnvironmentForTesting;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrUntypedException;
import com.google.protobuf.ByteString;
import java.util.Optional;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DigestWriter}. */
@RunWith(JUnit4.class)
public final class DigestWriterTest {

  @Test
  public void testComputePredeclaredInputHash_isDeterministicRegardlessOfRecordedRepoMappingOrder()
      throws Exception {
    Environment env =
        new AbstractSkyFunctionEnvironmentForTesting() {
          @Override
          protected ImmutableMap<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
              Iterable<? extends SkyKey> depKeys) {
            ImmutableMap.Builder<SkyKey, ValueOrUntypedException> builder = ImmutableMap.builder();
            for (SkyKey key : depKeys) {
              builder.put(
                  key,
                  ValueOrUntypedException.ofValueUntyped(
                      new EnvironmentVariableValue("dummy_val")));
            }
            return builder.buildOrThrow();
          }

          @Override
          public ExtendedEventHandler getListener() {
            return null;
          }
        };

    RepositoryName foo = RepositoryName.createUnvalidated("foo");
    RepositoryName bar = RepositoryName.createUnvalidated("bar");
    RepositoryName abc = RepositoryName.createUnvalidated("abc");
    RepositoryName xyz = RepositoryName.createUnvalidated("xyz");

    // Table 1 has (foo, attr1, bar) then (abc, attr2, xyz)
    ImmutableTable<RepositoryName, String, RepositoryName> table1 =
        ImmutableTable.<RepositoryName, String, RepositoryName>builder()
            .put(foo, "attr1", bar)
            .put(abc, "attr2", xyz)
            .buildOrThrow();

    // Table 2 has (abc, attr2, xyz) then (foo, attr1, bar)
    ImmutableTable<RepositoryName, String, RepositoryName> table2 =
        ImmutableTable.<RepositoryName, String, RepositoryName>builder()
            .put(abc, "attr2", xyz)
            .put(foo, "attr1", bar)
            .buildOrThrow();

    RepoRule rule1 = createRepoRule(table1);
    RepoRule rule2 = createRepoRule(table2);

    RepoDefinition def1 =
        new RepoDefinition(rule1, AttributeValues.create(Dict.empty()), "my_repo", null);
    RepoDefinition def2 =
        new RepoDefinition(rule2, AttributeValues.create(Dict.empty()), "my_repo", null);

    String hash1 = DigestWriter.computePredeclaredInputHash(env, def1, StarlarkSemantics.DEFAULT);
    String hash2 = DigestWriter.computePredeclaredInputHash(env, def2, StarlarkSemantics.DEFAULT);

    assertThat(hash1).isNotNull();
    assertThat(hash1).isEqualTo(hash2);
  }

  private RepoRule createRepoRule(
      ImmutableTable<RepositoryName, String, RepositoryName> repoMappingEntries) {
    RepoRule.Builder repoRuleBuilder = RepoRule.builder();
    repoRuleBuilder
        .configure(false)
        .doc(Optional.empty())
        .environ(ImmutableSet.of())
        .local(false)
        .remotable(false)
        .recordedRepoMappingEntries(repoMappingEntries)
        .impl(
            new StarlarkCallable() {
              @Override
              public String getName() {
                return "dummy_rule";
              }
            })
        .transitiveBzlDigest(ByteString.copyFromUtf8("dummy_digest"));
    repoRuleBuilder
        .idBuilder()
        .bzlFileLabel(Label.parseCanonicalUnchecked("//:test.bzl"))
        .ruleName("test");
    return repoRuleBuilder.build();
  }
}
