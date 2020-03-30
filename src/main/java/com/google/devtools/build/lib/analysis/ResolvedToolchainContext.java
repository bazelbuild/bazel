// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ToolchainException;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContext;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainContextApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import javax.annotation.Nullable;

/**
 * Represents the data needed for a specific target's use of toolchains and platforms, including
 * specific {@link ToolchainInfo} providers for each required toolchain type.
 */
@AutoValue
@Immutable
@ThreadSafe
public abstract class ResolvedToolchainContext implements ToolchainContextApi, ToolchainContext {

  /**
   * Finishes preparing the {@link ResolvedToolchainContext} by finding the specific toolchain
   * providers to be used for each toolchain type.
   */
  public static ResolvedToolchainContext load(
      ImmutableMap<RepositoryName, RepositoryName> repoMapping,
      UnloadedToolchainContext unloadedToolchainContext,
      String targetDescription,
      Iterable<ConfiguredTargetAndData> toolchainTargets)
      throws ToolchainException {

    ImmutableMap.Builder<ToolchainTypeInfo, ToolchainInfo> toolchains =
        new ImmutableMap.Builder<>();
    ImmutableList.Builder<TemplateVariableInfo> templateVariableProviders =
        new ImmutableList.Builder<>();
    for (ConfiguredTargetAndData target : toolchainTargets) {
      Label discoveredLabel;
      // Aliases are in toolchainTypeToResolved by the original alias label, not via the final
      // target's label.
      if (target.getConfiguredTarget() instanceof AliasConfiguredTarget) {
        discoveredLabel = ((AliasConfiguredTarget) target.getConfiguredTarget()).getOriginalLabel();
      } else {
        discoveredLabel = target.getConfiguredTarget().getLabel();
      }
      ToolchainTypeInfo toolchainType =
          unloadedToolchainContext.toolchainTypeToResolved().inverse().get(discoveredLabel);
      ToolchainInfo toolchainInfo = PlatformProviderUtils.toolchain(target.getConfiguredTarget());

      // If the toolchainType hadn't been resolved to an actual target, resolution would have
      // failed with an error much earlier. However, the target might still not be an actual
      // toolchain.
      if (toolchainType != null) {
        if (toolchainInfo != null) {
          toolchains.put(toolchainType, toolchainInfo);
        } else {
          throw new TargetNotToolchainException(toolchainType, discoveredLabel);
        }
      }

      // Find any template variables present for this toolchain.
      TemplateVariableInfo templateVariableInfo =
          target.getConfiguredTarget().get(TemplateVariableInfo.PROVIDER);
      if (templateVariableInfo != null) {
        templateVariableProviders.add(templateVariableInfo);
      }
    }

    return new AutoValue_ResolvedToolchainContext(
        // super:
        unloadedToolchainContext.executionPlatform(),
        unloadedToolchainContext.targetPlatform(),
        unloadedToolchainContext.requiredToolchainTypes(),
        unloadedToolchainContext.resolvedToolchainLabels(),
        // this:
        repoMapping,
        targetDescription,
        unloadedToolchainContext.requestedLabelToToolchainType(),
        toolchains.build(),
        templateVariableProviders.build());
  }

  /** Returns the repository mapping applied by the Starlark 'in' operator to string-form labels. */
  abstract ImmutableMap<RepositoryName, RepositoryName> repoMapping();

  /** Returns a description of the target being used, for error messaging. */
  abstract String targetDescription();

  /** Sets the map from requested {@link Label} to toolchain type provider. */
  abstract ImmutableMap<Label, ToolchainTypeInfo> requestedToolchainTypeLabels();

  abstract ImmutableMap<ToolchainTypeInfo, ToolchainInfo> toolchains();

  /** Returns the template variables that these toolchains provide. */
  public abstract ImmutableList<TemplateVariableInfo> templateVariableProviders();

  /**
   * Returns the toolchain for the given type, or {@code null} if the toolchain type was not
   * required in this context.
   */
  @Nullable
  public ToolchainInfo forToolchainType(Label toolchainTypeLabel) {
    ToolchainTypeInfo toolchainTypeInfo = requestedToolchainTypeLabels().get(toolchainTypeLabel);
    if (toolchainTypeInfo == null) {
      return null;
    }
    return toolchains().get(toolchainTypeInfo);
  }

  @Nullable
  public ToolchainInfo forToolchainType(ToolchainTypeInfo toolchainType) {
    return toolchains().get(toolchainType);
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<toolchain_context.resolved_labels: ");
    printer.append(
        toolchains().keySet().stream()
            .map(ToolchainTypeInfo::typeLabel)
            .map(Label::toString)
            .collect(joining(", ")));
    printer.append(">");
  }

  private static Label transformKey(
      Object key, ImmutableMap<RepositoryName, RepositoryName> repoMapping) throws EvalException {
    if (key instanceof Label) {
      return (Label) key;
    } else if (key instanceof ToolchainTypeInfo) {
      return ((ToolchainTypeInfo) key).typeLabel();
    } else if (key instanceof String) {
      try {
        return Label.parseAbsolute((String) key, repoMapping);
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
  public ToolchainInfo getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    Label toolchainTypeLabel = transformKey(key, repoMapping());

    if (!containsKey(semantics, key)) {
      // TODO(bazel-configurability): The list of available toolchain types is confusing in the
      // presence of aliases, since it only contains the actual label, not the alias passed to the
      // rule definition.
      throw Starlark.errorf(
          "In %s, toolchain type %s was requested but only types [%s] are configured",
          targetDescription(),
          toolchainTypeLabel,
          requiredToolchainTypes().stream()
              .map(ToolchainTypeInfo::typeLabel)
              .map(Label::toString)
              .collect(joining(", ")));
    }
    return forToolchainType(toolchainTypeLabel);
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    Label toolchainTypeLabel = transformKey(key, repoMapping());
    return requestedToolchainTypeLabels().containsKey(toolchainTypeLabel);
  }

  /**
   * Exception used when a toolchain type is required but the resolved target does not have
   * ToolchainInfo.
   */
  static final class TargetNotToolchainException extends ToolchainException {
    TargetNotToolchainException(ToolchainTypeInfo toolchainType, Label resolvedTargetLabel) {
      super(
          String.format(
              "toolchain type %s resolved to target %s, but that target does not provide "
                  + ToolchainInfo.SKYLARK_NAME,
              toolchainType.typeLabel(),
              resolvedTargetLabel));
    }
  }
}
