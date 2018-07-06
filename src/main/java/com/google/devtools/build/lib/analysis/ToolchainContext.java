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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skylarkbuildapi.ToolchainContextApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.Optional;
import java.util.Set;
import java.util.stream.StreamSupport;

/** Contains toolchain-related information needed for a {@link RuleContext}. */
@Immutable
@ThreadSafe
public class ToolchainContext implements ToolchainContextApi {

  public static ToolchainContext create(
      String targetDescription,
      PlatformInfo executionPlatform,
      PlatformInfo targetPlatform,
      Set<Label> requiredToolchains,
      ImmutableBiMap<Label, Label> resolvedLabels) {
    return new ToolchainContext(
        targetDescription,
        executionPlatform,
        targetPlatform,
        requiredToolchains,
        new ResolvedToolchainLabels(resolvedLabels));
  }

  /** The {@link PlatformInfo} describing where these toolchains can be executed. */
  private final PlatformInfo executionPlatform;

  /** The {@link PlatformInfo} describing the outputs of these toolchains. */
  private final PlatformInfo targetPlatform;

  /** Description of the target the toolchain context applies to, for use in error messages. */
  private final String targetDescription;

  /** The toolchain types that are required by the target. */
  private final ImmutableList<Label> requiredToolchains;

  /** Map from toolchain type labels to actual resolved toolchain labels. */
  private final ResolvedToolchainLabels resolvedToolchainLabels;

  /** Stores the actual ToolchainInfo provider for each toolchain type. */
  private ImmutableMap<Label, ToolchainInfo> toolchainProviders;

  private ToolchainContext(
      String targetDescription,
      PlatformInfo executionPlatform,
      PlatformInfo targetPlatform,
      Set<Label> requiredToolchains,
      ResolvedToolchainLabels resolvedToolchainLabels) {
    this.targetDescription = targetDescription;
    this.executionPlatform = executionPlatform;
    this.targetPlatform = targetPlatform;
    this.requiredToolchains = ImmutableList.copyOf(requiredToolchains);
    this.resolvedToolchainLabels = resolvedToolchainLabels;
    this.toolchainProviders = ImmutableMap.of();
  }

  public PlatformInfo executionPlatform() {
    return executionPlatform;
  }

  public PlatformInfo targetPlatform() {
    return targetPlatform;
  }

  public ImmutableList<Label> requiredToolchainTypes() {
    return requiredToolchains;
  }

  public void resolveToolchains(
      OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap) {
    if (!this.requiredToolchains.isEmpty()) {
      this.toolchainProviders = findToolchains(resolvedToolchainLabels, prerequisiteMap);
    }
  }

  public ImmutableSet<Label> resolvedToolchainLabels() {
    return resolvedToolchainLabels.getToolchainLabels();
  }

  /** Returns the {@link Label}s from the {@link NestedSet} that refer to toolchain dependencies. */
  public Set<Label> filterToolchainLabels(Iterable<Label> labels) {
    return StreamSupport.stream(labels.spliterator(), false)
        .filter(label -> resolvedToolchainLabels.isToolchainDependency(label))
        .collect(ImmutableSet.toImmutableSet());
  }

  /** Tracks the mapping from toolchain type label to the label of the actual resolved toolchain. */
  private static class ResolvedToolchainLabels {

    private final ImmutableBiMap<Label, Label> toolchainLabels;

    private ResolvedToolchainLabels(ImmutableBiMap<Label, Label> toolchainLabels) {
      this.toolchainLabels = toolchainLabels;
    }

    public Label getType(Label toolchainLabel) {
      return toolchainLabels.inverse().get(toolchainLabel);
    }

    public Label getResolvedToolchainLabel(Label toolchainType) {
      return toolchainLabels.get(toolchainType);
    }

    public ImmutableSet<Label> getToolchainLabels() {
      return toolchainLabels.values();
    }

    public boolean isToolchainDependency(Label label) {
      return toolchainLabels.containsValue(label);
    }
  }

  private static ImmutableMap<Label, ToolchainInfo> findToolchains(
      ResolvedToolchainLabels resolvedToolchainLabels,
      OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap) {
    // Find the prerequisites associated with PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR.
    Optional<Attribute> toolchainAttribute =
        prerequisiteMap
            .keys()
            .stream()
            .filter(attribute -> attribute != null)
            .filter(
                attribute -> attribute.getName().equals(PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR))
            .findFirst();
    Preconditions.checkState(
        toolchainAttribute.isPresent(),
        "No toolchains attribute found while loading resolved toolchains");

    ImmutableMap.Builder<Label, ToolchainInfo> toolchains = new ImmutableMap.Builder<>();
    for (ConfiguredTargetAndData target : prerequisiteMap.get(toolchainAttribute.get())) {
      Label discoveredLabel = target.getTarget().getLabel();
      Label toolchainType = resolvedToolchainLabels.getType(discoveredLabel);
      if (toolchainType != null) {
        ToolchainInfo toolchainInfo = PlatformProviderUtils.toolchain(target.getConfiguredTarget());
        toolchains.put(toolchainType, toolchainInfo);
      }
    }

    return toolchains.build();
  }

  // Implement SkylarkValue and SkylarkIndexable.

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<toolchain_context.resolved_labels: ");
    printer.append(
        toolchainProviders.keySet().stream().map(key -> key.toString()).collect(joining(", ")));
    printer.append(">");
  }

  private Label transformKey(Object key, Location loc) throws EvalException {
    if (key instanceof Label) {
      Label toolchainType = (Label) key;
      return toolchainType;
    } else if (key instanceof String) {
      Label toolchainType = null;
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
  public ToolchainInfo getIndex(Object key, Location loc) throws EvalException {
    Label toolchainType = transformKey(key, loc);

    if (!requiredToolchains.contains(toolchainType)) {
      throw new EvalException(
          loc,
          String.format(
              "In %s, toolchain type %s was requested but only types [%s] are configured",
              targetDescription,
              toolchainType,
              requiredToolchains
                  .stream()
                  .map(toolchain -> toolchain.toString())
                  .collect(joining())));
    }
    return toolchainProviders.get(toolchainType);
  }

  /** Returns the toolchain for the given type */
  public ToolchainInfo forToolchainType(Label toolchainType) {
    return toolchainProviders.get(toolchainType);
  }

  @Override
  public boolean containsKey(Object key, Location loc) throws EvalException {
    Label toolchainType = transformKey(key, loc);
    return toolchainProviders.containsKey(toolchainType);
  }
}
