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
package com.google.devtools.build.lib.analysis.starlark;

import static com.google.devtools.build.lib.analysis.ToolchainCollection.DEFAULT_EXEC_GROUP_NAME;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ResolvedToolchainContext;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ExecGroupCollectionApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ToolchainContextApi;
import java.util.List;
import java.util.stream.Collectors;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkIndexable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Identifier;

/**
 * A {@link StarlarkIndexable} collection of resolved toolchain contexts that can be exposed to
 * starlark.
 */
@AutoValue
public abstract class StarlarkExecGroupCollection implements ExecGroupCollectionApi {

  /**
   * Returns a new {@link StarlarkExecGroupCollection} backed by the given {@code
   * toolchainCollection}.
   */
  public static StarlarkExecGroupCollection create(
      ToolchainCollection<ResolvedToolchainContext> toolchainCollection) {
    return new AutoValue_StarlarkExecGroupCollection(toolchainCollection);
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
   * This creates a new {@link StarlarkExecGroupContext} object every time this is called. This
   * seems better than pre-creating and storing all {@link StarlarkExecGroupContext}s since they're
   * just thin wrappers around {@link ResolvedToolchainContext} objects.
   */
  @Override
  public StarlarkExecGroupContext getIndex(StarlarkSemantics semantics, Object key)
      throws EvalException {
    String execGroup = castGroupName(key);
    if (!containsKey(semantics, key)) {
      throw Starlark.errorf(
          "In %s, unrecognized exec group '%s' requested. Available exec groups: [%s]",
          toolchainCollection().getDefaultToolchainContext().targetDescription(),
          execGroup,
          String.join(", ", getScrubbedExecGroups()));
    }
    ToolchainContextApi toolchainContext = toolchainCollection().getToolchainContext(execGroup);
    return new AutoValue_StarlarkExecGroupCollection_StarlarkExecGroupContext(toolchainContext);
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
  @AutoValue
  public abstract static class StarlarkExecGroupContext implements ExecGroupContextApi {
    @Override
    public abstract ToolchainContextApi toolchains();

    @Override
    public void repr(Printer printer) {
      printer.append("<exec_group_context>");
    }
  }
}
