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
package com.google.devtools.build.lib.analysis.starlark;

import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ToolchainContextApi;
import java.util.function.Function;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/**
 * An implementation of ToolchainContextApi that can better handle converting strings into Labels.
 */
@AutoValue
public abstract class StarlarkToolchainContext implements ToolchainContextApi {

  public static final ToolchainContextApi TOOLCHAINS_NOT_VALID =
      new ToolchainContextApi() {
        @Override
        public Object getIndex(
            StarlarkThread starlarkThread, StarlarkSemantics semantics, Object key)
            throws EvalException {
          throw Starlark.errorf("Toolchains are not valid in this context");
        }

        @Override
        public boolean containsKey(
            StarlarkThread starlarkThread, StarlarkSemantics semantics, Object key) {
          return false;
        }
      };

  public static ToolchainContextApi create(
      String targetDescription,
      Function<Label, ToolchainInfo> resolveToolchainInfoFunc,
      ImmutableSet<Label> resolvedToolchainTypeLabels) {
    Preconditions.checkNotNull(targetDescription);
    Preconditions.checkNotNull(resolveToolchainInfoFunc);
    Preconditions.checkNotNull(resolvedToolchainTypeLabels);

    return new AutoValue_StarlarkToolchainContext(
        targetDescription, resolveToolchainInfoFunc, resolvedToolchainTypeLabels);
  }

  protected abstract String targetDescription();

  protected abstract Function<Label, ToolchainInfo> resolveToolchainInfoFunc();

  protected abstract ImmutableSet<Label> resolvedToolchainTypeLabels();

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<toolchain_context.resolved_labels: ");
    printer.append(
        resolvedToolchainTypeLabels().stream().map(Label::toString).collect(joining(", ")));
    printer.append(">");
  }

  private Label transformKey(StarlarkThread starlarkThread, Object key) throws EvalException {
    if (key instanceof Label label) {
      return label;
    } else if (key instanceof ToolchainTypeInfo toolchainTypeInfo) {
      return toolchainTypeInfo.typeLabel();
    } else if (key instanceof String) {
      try {
        LabelConverter converter = LabelConverter.forBzlEvaluatingThread(starlarkThread);
        return converter.convert((String) key);
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf("Unable to parse toolchain label '%s': %s", key, e.getMessage());
      }
    } else {
      throw Starlark.errorf(
          "Toolchains only supports indexing by toolchain type, got %s instead",
          Starlark.type(key));
    }
  }


  @Override
  public StarlarkValue getIndex(
      StarlarkThread starlarkThread, StarlarkSemantics semantics, Object key) throws EvalException {
    Label toolchainTypeLabel = transformKey(starlarkThread, key);

    if (!containsKey(starlarkThread, semantics, key)) {
      // TODO(bazel-configurability): The list of available toolchain types is confusing in the
      // presence of aliases, since it only contains the actual label, not the alias passed to the
      // rule definition.
      throw Starlark.errorf(
          "In %s, toolchain type %s was requested but only types [%s] are configured",
          targetDescription(),
          toolchainTypeLabel,
          resolvedToolchainTypeLabels().stream().map(Label::toString).collect(joining(", ")));
    }
    ToolchainInfo toolchainInfo = resolveToolchainInfoFunc().apply(toolchainTypeLabel);
    if (toolchainInfo == null) {
      return Starlark.NONE;
    }
    return toolchainInfo;
  }


  @Override
  public boolean containsKey(StarlarkThread starlarkThread, StarlarkSemantics semantics, Object key)
      throws EvalException {
    Label toolchainTypeLabel = transformKey(starlarkThread, key);
    return resolvedToolchainTypeLabels().contains(toolchainTypeLabel);
  }
}
