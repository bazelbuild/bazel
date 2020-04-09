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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A wrapper class for a map of exec_group names to their relevent ToolchainContext.
 *
 * @param <T> any class that extends ToolchainCollection so this can be used, e.g., both before and
 *     after toolchain resolution.
 */
public class ToolchainCollection<T extends ToolchainContext> {
  @VisibleForTesting public static final String DEFAULT_EXEC_GROUP_NAME = "default_exec_group";

  /** A map of execution group names to toolchain contexts. */
  private final ImmutableMap<String, T> toolchainContexts;

  private ToolchainCollection(Map<String, T> contexts) {
    Preconditions.checkArgument(contexts.containsKey(DEFAULT_EXEC_GROUP_NAME));
    toolchainContexts = ImmutableMap.copyOf(contexts);
  }

  /** Builder for ToolchainCollection. */
  public static class Builder<T extends ToolchainContext> {
    private final Map<String, T> toolchainContexts = new HashMap<>();

    public ToolchainCollection<T> build() {
      return new ToolchainCollection<>(toolchainContexts);
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

  public T getDefaultToolchainContext() {
    return toolchainContexts.get(DEFAULT_EXEC_GROUP_NAME);
  }

  public T getToolchainContext(String execGroup) {
    return toolchainContexts.get(execGroup);
  }

  public ImmutableSet<Label> getRequiredToolchains() {
    Set<Label> requiredToolchains = new HashSet<>();
    for (T context : toolchainContexts.values()) {
      requiredToolchains.addAll(context.resolvedToolchainLabels());
    }
    return ImmutableSet.copyOf(requiredToolchains);
  }

  public ToolchainCollection<ToolchainContext> asToolchainContexts() {
    return new ToolchainCollection<>(ImmutableMap.copyOf(toolchainContexts));
  }

  public ImmutableMap<String, T> getContextMap() {
    return toolchainContexts;
  }
}
