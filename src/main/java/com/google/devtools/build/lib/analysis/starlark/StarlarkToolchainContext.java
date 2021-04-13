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
import com.google.devtools.build.lib.analysis.ResolvedToolchainContext;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.BuildType.LabelConversionContext;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ToolchainContextApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/**
 * An implementation of ToolchainContextApi that can better handle converting strings into Labels.
 */
@AutoValue
public abstract class StarlarkToolchainContext implements ToolchainContextApi {

  private static final ToolchainContextApi NO_OP =
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

  public static ToolchainContextApi create(@Nullable ResolvedToolchainContext toolchainContext) {
    if (toolchainContext == null) {
      return NO_OP;
    }

    return new AutoValue_StarlarkToolchainContext(toolchainContext);
  }

  protected abstract ResolvedToolchainContext toolchainContext();

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<toolchain_context.resolved_labels: ");
    printer.append(
        toolchainContext().toolchains().keySet().stream()
            .map(ToolchainTypeInfo::typeLabel)
            .map(Label::toString)
            .collect(joining(", ")));
    printer.append(">");
  }

  private Label transformKey(StarlarkThread starlarkThread, Object key) throws EvalException {
    if (key instanceof Label) {
      return (Label) key;
    } else if (key instanceof ToolchainTypeInfo) {
      return ((ToolchainTypeInfo) key).typeLabel();
    } else if (key instanceof String) {
      try {
        BazelStarlarkContext bazelStarlarkContext = BazelStarlarkContext.from(starlarkThread);
        LabelConversionContext context =
            new LabelConversionContext(
                BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(starlarkThread))
                    .label(),
                bazelStarlarkContext.getRepoMapping(),
                bazelStarlarkContext.getConvertedLabelsInPackage());
        return context.convert((String) key);
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
  public ToolchainInfo getIndex(
      StarlarkThread starlarkThread, StarlarkSemantics semantics, Object key) throws EvalException {
    Label toolchainTypeLabel = transformKey(starlarkThread, key);

    if (!containsKey(starlarkThread, semantics, key)) {
      // TODO(bazel-configurability): The list of available toolchain types is confusing in the
      // presence of aliases, since it only contains the actual label, not the alias passed to the
      // rule definition.
      throw Starlark.errorf(
          "In %s, toolchain type %s was requested but only types [%s] are configured",
          toolchainContext().targetDescription(),
          toolchainTypeLabel,
          toolchainContext().requiredToolchainTypes().stream()
              .map(ToolchainTypeInfo::typeLabel)
              .map(Label::toString)
              .collect(joining(", ")));
    }
    return toolchainContext().forToolchainType(toolchainTypeLabel);
  }

  @Override
  public boolean containsKey(StarlarkThread starlarkThread, StarlarkSemantics semantics, Object key)
      throws EvalException {
    Label toolchainTypeLabel = transformKey(starlarkThread, key);
    return toolchainContext().requestedToolchainTypeLabels().containsKey(toolchainTypeLabel);
  }
}
