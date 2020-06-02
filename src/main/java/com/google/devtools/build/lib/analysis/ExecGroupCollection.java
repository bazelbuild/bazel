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

import static com.google.devtools.build.lib.analysis.ToolchainCollection.DEFAULT_EXEC_GROUP_NAME;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ExecGroupCollectionApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkIndexable;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A {@link StarlarkIndexable} collection of resolved toolchain contexts that can be exposed to
 * starlark.
 */
@AutoValue
public abstract class ExecGroupCollection implements ExecGroupCollectionApi {

  /** Returns a new {@link ExecGroupCollection} backed by the given {@code toolchainCollection}. */
  public static ExecGroupCollection create(
      ToolchainCollection<ResolvedToolchainContext> toolchainCollection) {
    return new AutoValue_ExecGroupCollection(toolchainCollection);
  }

  protected abstract ToolchainCollection<ResolvedToolchainContext> toolchainCollection();

  @VisibleForTesting
  public ImmutableMap<String, ResolvedToolchainContext> getToolchainCollectionForTesting() {
    return toolchainCollection().getContextMap();
  }

  public static boolean isValidGroupName(String execGroupName) {
    return !execGroupName.equals(DEFAULT_EXEC_GROUP_NAME) && Identifier.isValid(execGroupName);
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    String group = castGroupName(key);
    return !DEFAULT_EXEC_GROUP_NAME.equals(group)
        && toolchainCollection().getExecGroups().contains(group);
  }

  /**
   * This creates a new {@link ExecGroupContext} object every time this is called. This seems better
   * than pre-creating and storing all {@link ExecGroupContext}s since they're just thin wrappers
   * around {@link ResolvedToolchainContext} objects.
   */
  @Override
  public ExecGroupContext getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    String execGroup = castGroupName(key);
    if (!containsKey(semantics, key)) {
      throw Starlark.errorf(
          "In %s, unrecognized exec group '%s' requested. Available exec groups: [%s]",
          toolchainCollection().getDefaultToolchainContext().targetDescription(),
          execGroup,
          String.join(", ", getScrubbedExecGroups()));
    }
    return new ExecGroupContext(toolchainCollection().getToolchainContext(execGroup));
  }

  private static String castGroupName(Object key) throws EvalException {
    if (!(key instanceof String)) {
      throw Starlark.errorf(
          "exec groups only support indexing by exec group name, got %s of type %s instead",
          Starlark.repr(key), Starlark.type(key));
    }
    return (String) key;
  }

  @Override
  public void repr(Printer printer) {
    printer
        .append("<ctx.exec_groups: ")
        .append(String.join(", ", getScrubbedExecGroups()))
        .append(">");
  }

  private List<String> getScrubbedExecGroups() {
    return toolchainCollection().getExecGroups().stream()
        .filter(group -> !DEFAULT_EXEC_GROUP_NAME.equals(group))
        .sorted()
        .collect(Collectors.toList());
  }

  /**
   * The starlark object that is returned by ctx.exec_groups[<name>]. Gives information about that
   * exec group.
   */
  public static class ExecGroupContext implements ExecGroupContextApi {
    ResolvedToolchainContext resolvedToolchainContext;

    private ExecGroupContext(ResolvedToolchainContext resolvedToolchainContext) {
      this.resolvedToolchainContext = resolvedToolchainContext;
    }

    @Override
    public ResolvedToolchainContext toolchains() {
      return resolvedToolchainContext;
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<exec_group_context>");
    }
  }
}
