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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import java.util.Optional;

/**
 * A class holding information about the attributes of a repository rule and where the rule class is
 * defined.
 */
@AutoValue
public abstract class RepoSpec {

  /**
   * The label string for the bzl file this repository rule is defined in, empty for native rule.
   */
  public abstract Optional<String> bzlFile();

  public abstract String ruleClassName();

  public abstract ImmutableMap<String, Object> attributes();

  public static Builder builder() {
    return new AutoValue_RepoSpec.Builder();
  }

  /** The builder for {@link RepoSpec} */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setBzlFile(String bzlFile);

    public abstract Builder setRuleClassName(String name);

    public abstract Builder setAttributes(ImmutableMap<String, Object> attributes);

    public abstract RepoSpec build();
  }

  public boolean isNativeRepoRule() {
    return !bzlFile().isPresent();
  }

  /**
   * Return a string representing the rule class eg. Native repo rule: local_repository, Starlark
   * repo rule: //:repo.bzl%my_repo
   */
  public String getRuleClass() {
    return bzlFile().map(f -> f + "%").orElse("") + ruleClassName();
  }
}
