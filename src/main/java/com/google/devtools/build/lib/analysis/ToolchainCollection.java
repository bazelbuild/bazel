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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.HashMap;
import java.util.Map;

/**
 * A wrapper class for a map of exec_group names to their relevant ToolchainContext.
 *
 * @param <T> any class that extends ToolchainContext. This generic allows ToolchainCollection to be
 *     used, e.g., both before and after toolchain resolution.
 */
@AutoValue
public abstract class ToolchainCollection<T extends ToolchainContext> {

  // This is intentionally a string that would fail {@code Identifier.isValid} so that
  // users can't create a group with the same name.
  @VisibleForTesting public static final String DEFAULT_EXEC_GROUP_NAME = "default-exec-group";

  /** A map of execution group names to toolchain contexts. */
  public abstract ImmutableMap<String, T> getContextMap();

  public T getDefaultToolchainContext() {
    return getContextMap().get(DEFAULT_EXEC_GROUP_NAME);
  }

  public boolean hasToolchainContext(String execGroup) {
    return getContextMap().containsKey(execGroup);
  }

  public T getToolchainContext(String execGroup) {
    return getContextMap().get(execGroup);
  }

  public ImmutableSet<Label> getResolvedToolchains() {
    return getContextMap().values().stream()
        .flatMap(c -> c.resolvedToolchainLabels().stream())
        .collect(toImmutableSet());
  }

  public ImmutableSet<String> getExecGroups() {
    return getContextMap().keySet();
  }

  @SuppressWarnings("unchecked")
  public ToolchainCollection<ToolchainContext> asToolchainContexts() {
    return (ToolchainCollection<ToolchainContext>) this;
  }

  /** Returns a new builder for {@link ToolchainCollection} instances. */
  public static <T extends ToolchainContext> Builder<T> builder() {
    return new Builder<T>();
  }

  /** Builder for ToolchainCollection. */
  public static final class Builder<T extends ToolchainContext> {
    // This is not immutable so that we can check for duplicate keys easily.
    private final Map<String, T> toolchainContexts = new HashMap<>();

    public ToolchainCollection<T> build() {
      Preconditions.checkArgument(toolchainContexts.containsKey(DEFAULT_EXEC_GROUP_NAME));
      return new AutoValue_ToolchainCollection<T>(ImmutableMap.copyOf(toolchainContexts));
    }

    public void addContext(String execGroup, T context) {
      Preconditions.checkArgument(
          !toolchainContexts.containsKey(execGroup),
          "Duplicate add of '%s' exec group to toolchain collection.",
          execGroup);
      toolchainContexts.put(execGroup, context);
    }

    public Builder<T> addDefaultContext(T context) {
      addContext(DEFAULT_EXEC_GROUP_NAME, context);
      return this;
    }
  }
}
