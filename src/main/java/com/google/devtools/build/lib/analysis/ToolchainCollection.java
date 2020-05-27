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
public class ToolchainCollection<T extends ToolchainContext> {

  // This is intentionally a string that would fail {@code Identifier.isValid} so that
  // users can't create a group with the same name.
  @VisibleForTesting public static final String DEFAULT_EXEC_GROUP_NAME = "default-exec-group";

  /** A map of execution group names to toolchain contexts. */
  private final ImmutableMap<String, T> toolchainContexts;

  private ToolchainCollection(Map<String, T> contexts) {
    Preconditions.checkArgument(contexts.containsKey(DEFAULT_EXEC_GROUP_NAME));
    toolchainContexts = ImmutableMap.copyOf(contexts);
  }

  ToolchainCollection(ToolchainCollection<T> toCopy) {
    toolchainContexts = ImmutableMap.copyOf(toCopy.getContextMap());
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

  T getDefaultToolchainContext() {
    return toolchainContexts.get(DEFAULT_EXEC_GROUP_NAME);
  }

  boolean hasToolchainContext(String execGroup) {
    return toolchainContexts.containsKey(execGroup);
  }

  public T getToolchainContext(String execGroup) {
    return toolchainContexts.get(execGroup);
  }

  public ImmutableSet<Label> getResolvedToolchains() {
    return toolchainContexts.values().stream()
        .flatMap(c -> c.resolvedToolchainLabels().stream())
        .collect(toImmutableSet());
  }

  ImmutableSet<String> getExecGroups() {
    return toolchainContexts.keySet();
  }

  public ToolchainCollection<ToolchainContext> asToolchainContexts() {
    return new ToolchainCollection<>(ImmutableMap.copyOf(toolchainContexts));
  }

  public ImmutableMap<String, T> getContextMap() {
    return toolchainContexts;
  }
}
