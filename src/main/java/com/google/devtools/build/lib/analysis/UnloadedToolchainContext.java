// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ToolchainException;
import java.util.Set;

/**
 * Represents the state of toolchain resolution once the specific required toolchains have been
 * determined, but before the toolchain dependencies have been resolved.
 */
@AutoValue
public abstract class UnloadedToolchainContext implements ToolchainContext {

  static Builder builder() {
    return new AutoValue_UnloadedToolchainContext.Builder();
  }

  /** Builder class to help create the {@link UnloadedToolchainContext}. */
  @AutoValue.Builder
  public interface Builder {
    /** Sets a description of the target being used, for error messaging. */
    Builder setTargetDescription(String targetDescription);

    /** Sets the selected execution platform that these toolchains use. */
    Builder setExecutionPlatform(PlatformInfo executionPlatform);

    /** Sets the target platform that these toolchains generate output for. */
    Builder setTargetPlatform(PlatformInfo targetPlatform);

    /** Sets the toolchain types that were requested. */
    Builder setRequiredToolchainTypes(Set<ToolchainTypeInfo> requiredToolchainTypes);

    Builder setToolchainTypeToResolved(
        ImmutableBiMap<ToolchainTypeInfo, Label> toolchainTypeToResolved);

    UnloadedToolchainContext build();
  }

  /** Returns a description of the target being used, for error messaging. */
  abstract String targetDescription();

  /** The map of toolchain type to resolved toolchain to be used. */
  abstract ImmutableBiMap<ToolchainTypeInfo, Label> toolchainTypeToResolved();

  @Override
  public ImmutableSet<Label> resolvedToolchainLabels() {
    return toolchainTypeToResolved().values();
  }

  /**
   * Finishes preparing the {@link ResolvedToolchainContext} by finding the specific toolchain
   * providers to be used for each toolchain type.
   */
  public ResolvedToolchainContext load(Iterable<ConfiguredTargetAndData> toolchainTargets)
      throws ToolchainException {

    ResolvedToolchainContext.Builder toolchainContext =
        ResolvedToolchainContext.builder()
            .setTargetDescription(targetDescription())
            .setExecutionPlatform(executionPlatform())
            .setTargetPlatform(targetPlatform())
            .setRequiredToolchainTypes(requiredToolchainTypes())
            .setResolvedToolchainLabels(resolvedToolchainLabels());

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
      ToolchainTypeInfo toolchainType = toolchainTypeToResolved().inverse().get(discoveredLabel);
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

  /**
   * Exception used when a toolchain type is required but the resolved target does not have
   * ToolchainInfo.
   */
  static final class TargetNotToolchainException extends ToolchainException {
    TargetNotToolchainException(ToolchainTypeInfo toolchainType, Label resolvedTargetLabel) {
      super(
          String.format(
              "toolchain type %s resolved to target %s, but that target does not provide"
                  + " ToolchainInfo",
              toolchainType.typeLabel(), resolvedTargetLabel));
    }
  }
}
