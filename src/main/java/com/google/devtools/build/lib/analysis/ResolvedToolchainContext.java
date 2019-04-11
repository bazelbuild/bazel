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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ToolchainException;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContext;
import com.google.devtools.build.lib.skylarkbuildapi.ToolchainContextApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import java.util.Optional;
import java.util.Set;
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
      UnloadedToolchainContext unloadedToolchainContext,
      String targetDescription,
      Iterable<ConfiguredTargetAndData> toolchainTargets)
      throws ToolchainException {

    ResolvedToolchainContext.Builder toolchainContext =
        new AutoValue_ResolvedToolchainContext.Builder()
            .setTargetDescription(targetDescription)
            .setExecutionPlatform(unloadedToolchainContext.executionPlatform())
            .setTargetPlatform(unloadedToolchainContext.targetPlatform())
            .setRequiredToolchainTypes(unloadedToolchainContext.requiredToolchainTypes())
            .setResolvedToolchainLabels(unloadedToolchainContext.resolvedToolchainLabels());

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

    return toolchainContext
        .setToolchains(toolchains.build())
        .setTemplateVariableProviders(templateVariableProviders.build())
        .build();
  }

  /** Builder interface to help create new instances of {@link ResolvedToolchainContext}. */
  @AutoValue.Builder
  interface Builder {
    /** Sets a description of the target being used, for error messaging. */
    Builder setTargetDescription(String targetDescription);

    /** Sets the selected execution platform that these toolchains use. */
    Builder setExecutionPlatform(PlatformInfo executionPlatform);

    /** Sets the target platform that these toolchains generate output for. */
    Builder setTargetPlatform(PlatformInfo targetPlatform);

    /** Sets the toolchain types that were requested. */
    Builder setRequiredToolchainTypes(Set<ToolchainTypeInfo> requiredToolchainTypes);

    /** Sets the map from toolchain type to toolchain provider. */
    Builder setToolchains(ImmutableMap<ToolchainTypeInfo, ToolchainInfo> toolchains);

    /** Sets the template variables that these toolchains provide. */
    Builder setTemplateVariableProviders(ImmutableList<TemplateVariableInfo> providers);

    /** Sets the labels of the specific toolchains being used. */
    Builder setResolvedToolchainLabels(ImmutableSet<Label> resolvedToolchainLabels);

    /** Returns a new {@link ResolvedToolchainContext}. */
    ResolvedToolchainContext build();
  }

  /** Returns a description of the target being used, for error messaging. */
  abstract String targetDescription();

  abstract ImmutableMap<ToolchainTypeInfo, ToolchainInfo> toolchains();

  /** Returns the template variables that these toolchains provide. */
  public abstract ImmutableList<TemplateVariableInfo> templateVariableProviders();

  /**
   * Returns the toolchain for the given type, or {@code null} if the toolchain type was not
   * required in this context.
   */
  @Nullable
  public ToolchainInfo forToolchainType(Label toolchainTypeLabel) {
    Optional<ToolchainTypeInfo> toolchainType =
        toolchains().keySet().stream()
            .filter(info -> info.typeLabel().equals(toolchainTypeLabel))
            .findFirst();
    if (toolchainType.isPresent()) {
      return forToolchainType(toolchainType.get());
    } else {
      return null;
    }
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
  public void repr(SkylarkPrinter printer) {
    printer.append("<toolchain_context.resolved_labels: ");
    printer.append(
        toolchains().keySet().stream()
            .map(ToolchainTypeInfo::typeLabel)
            .map(Label::toString)
            .collect(joining(", ")));
    printer.append(">");
  }

  private Label transformKey(Object key, Location loc) throws EvalException {
    if (key instanceof Label) {
      return (Label) key;
    } else if (key instanceof ToolchainTypeInfo) {
      return ((ToolchainTypeInfo) key).typeLabel();
    } else if (key instanceof String) {
      Label toolchainType;
      String rawLabel = (String) key;
      try {
        toolchainType = Label.parseAbsolute(rawLabel, ImmutableMap.of());
      } catch (LabelSyntaxException e) {
        throw new EvalException(
            loc, String.format("Unable to parse toolchain %s: %s", rawLabel, e.getMessage()), e);
      }
      return toolchainType;
    } else {
      throw new EvalException(
          loc,
          String.format(
              "Toolchains only supports indexing by toolchain type, got %s instead",
              EvalUtils.getDataTypeName(key)));
    }
  }

  @Override
  public ToolchainInfo getIndex(Object key, Location loc, StarlarkContext context)
      throws EvalException {
    Label toolchainTypeLabel = transformKey(key, loc);

    if (!containsKey(key, loc, context)) {
      throw new EvalException(
          loc,
          String.format(
              "In %s, toolchain type %s was requested but only types [%s] are configured",
              targetDescription(),
              toolchainTypeLabel,
              requiredToolchainTypes().stream()
                  .map(ToolchainTypeInfo::typeLabel)
                  .map(Label::toString)
                  .collect(joining(", "))));
    }
    return forToolchainType(toolchainTypeLabel);
  }

  @Override
  public boolean containsKey(Object key, Location loc, StarlarkContext context)
      throws EvalException {
    Label toolchainTypeLabel = transformKey(key, loc);
    Optional<Label> matching =
        toolchains().keySet().stream()
            .map(ToolchainTypeInfo::typeLabel)
            .filter(label -> label.equals(toolchainTypeLabel))
            .findAny();
    return matching.isPresent();
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
