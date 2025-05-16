// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.base.Preconditions.checkNotNull;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import java.util.HashSet;
import java.util.Map;

/** A helper interface for printing debug messages from toolchain resolution. */
public sealed interface ToolchainResolutionDebugPrinter {

  static ToolchainResolutionDebugPrinter create(boolean debug, SkyFunction.Environment env)
      throws InterruptedException {
    if (!debug) {
      return new NoopPrinter();
    }
    var mainRepositoryMapping =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    checkNotNull(
        mainRepositoryMapping,
        "expected main repository mapping to have been computed when resolving toolchains");
    return env.getState(() -> new EventHandlerImpl(mainRepositoryMapping.repositoryMapping()));
  }

  boolean debugEnabled();

  void startDebugging(
      EventHandler eventHandler,
      Label targetPlatform,
      String configurationId,
      ImmutableSet<ToolchainTypeRequirement> toolchainTypeRequirements);

  /** Report on an execution platform that was skipped due to constraint mismatches. */
  void reportRemovedExecutionPlatform(
      EventHandler eventHandler,
      Label label,
      ImmutableList<ConstraintValueInfo> missingConstraints);

  void reportRejectedExecutionPlatforms(
      EventHandler eventHandler, ImmutableMap<Label, String> rejectedExecutionPlatforms);

  /** Report on which toolchains were selected. */
  void finishDebugging(
      EventHandler eventHandler,
      Label targetPlatform,
      String configurationId,
      Label executionPlatform,
      ImmutableSetMultimap<ToolchainTypeInfo, Label> toolchainTypeToResolved);

  /** A do-nothing implementation for when debug messages are suppressed. */
  final class NoopPrinter implements ToolchainResolutionDebugPrinter {

    private NoopPrinter() {}

    @Override
    public boolean debugEnabled() {
      return false;
    }

    @Override
    public void startDebugging(
        EventHandler eventHandler,
        Label targetPlatform,
        String configurationId,
        ImmutableSet<ToolchainTypeRequirement> toolchainTypeRequirements) {}

    @Override
    public void reportRemovedExecutionPlatform(
        EventHandler eventHandler,
        Label label,
        ImmutableList<ConstraintValueInfo> missingConstraints) {}

    @Override
    public void reportRejectedExecutionPlatforms(
        EventHandler eventHandler, ImmutableMap<Label, String> rejectedExecutionPlatforms) {}

    @Override
    public void finishDebugging(
        EventHandler eventHandler,
        Label targetPlatform,
        String configurationId,
        Label executionPlatform,
        ImmutableSetMultimap<ToolchainTypeInfo, Label> toolchainTypeToResolved) {}
  }

  /** Implement debug printing using the {@link ExtendedEventHandler}. */
  final class EventHandlerImpl
      implements ToolchainResolutionDebugPrinter, SkyFunction.Environment.SkyKeyComputeState {
    private final RepositoryMapping mainRepositoryMapping;

    public EventHandlerImpl(RepositoryMapping mainRepositoryMapping) {
      this.mainRepositoryMapping = mainRepositoryMapping;
    }

    @Override
    public boolean debugEnabled() {
      return true;
    }

    private final HashSet<Event> handledEvents = new HashSet<>();

    @FormatMethod
    private void debugMessage(
        EventHandler eventHandler, @FormatString String template, Object... args) {
      // Avoid showing the same message multiple times due to Skyframe restarts. This relies on each
      // message being unique within the lifetime of a single instance.
      // Note that deduplication is best-effort and may fail if the cache is dropped due to memory
      // pressure.
      var event = Event.info("ToolchainResolution: " + String.format(template, args));
      if (handledEvents.add(event)) {
        eventHandler.handle(event);
      }
    }

    @Override
    public void startDebugging(
        EventHandler eventHandler,
        Label targetPlatform,
        String configurationId,
        ImmutableSet<ToolchainTypeRequirement> toolchainTypeRequirements) {
      debugMessage(
          eventHandler,
          "Starting for target platform %s (config %s)%s",
          targetPlatform.getShorthandDisplayForm(mainRepositoryMapping),
          configurationId,
          toolchainTypeRequirements.isEmpty()
              ? ""
              : String.format(
                  " and types %s",
                  toolchainTypeRequirements.stream()
                      .map(
                          type ->
                              type.toolchainType().getShorthandDisplayForm(mainRepositoryMapping))
                      .collect(joining(", "))));
    }

    @Override
    public void reportRemovedExecutionPlatform(
        EventHandler eventHandler,
        Label label,
        ImmutableList<ConstraintValueInfo> missingConstraints) {
      // TODO: jcater - Make this one line listing all constraints.
      for (ConstraintValueInfo constraint : missingConstraints) {
        // The value for this setting is not present in the platform, or doesn't match the
        // expected value.
        debugMessage(
            eventHandler,
            "Skipping execution platform %s from available execution platforms, it is missing constraint %s",
            label,
            constraint.label());
      }
    }

    @Override
    public void reportRejectedExecutionPlatforms(
        EventHandler eventHandler, ImmutableMap<Label, String> rejectedExecutionPlatforms) {
      if (!rejectedExecutionPlatforms.isEmpty()) {
        for (Map.Entry<Label, String> entry : rejectedExecutionPlatforms.entrySet()) {
          Label toolchainLabel = entry.getKey();
          String message = entry.getValue();
          debugMessage(eventHandler, "Rejected execution platform %s; %s", toolchainLabel, message);
        }
      }
    }

    @Override
    public void finishDebugging(
        EventHandler eventHandler,
        Label targetPlatform,
        String configurationId,
        Label executionPlatform,
        ImmutableSetMultimap<ToolchainTypeInfo, Label> toolchainTypeToResolved) {
      String selectedToolchains =
          toolchainTypeToResolved.entries().stream()
              .map(
                  e ->
                      String.format(
                          "type %s -> toolchain %s",
                          e.getKey().typeLabel().getShorthandDisplayForm(mainRepositoryMapping),
                          e.getValue()))
              .collect(joining(", "));
      debugMessage(
          eventHandler,
          "Finished for target platform %s (config %s): execution platform %s%s%s",
          targetPlatform.getShorthandDisplayForm(mainRepositoryMapping),
          configurationId,
          executionPlatform.getShorthandDisplayForm(mainRepositoryMapping),
          selectedToolchains.isEmpty() ? "" : ", ",
          selectedToolchains);
    }
  }
}
