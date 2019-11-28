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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;

/**
 * A representation of command line for C++ actions ({@link CppCompileAction}, {@link
 * CppLinkAction}, {@link LtoBackendAction}...).
 *
 * <p>This class is an adapter from {@link FeatureConfiguration} and {@link CcToolchainVariables} to
 * {@link CommandLine}. Using this we're able to pass C++ command lines to {@link
 * com.google.devtools.build.lib.analysis.actions.SpawnAction} or to give access to C++ command
 * lines to Starlark aspects.
 *
 * <p>See {@link CompileCommandLine#getArguments} for inline subclass of CommandLine that does all
 * of this plus some more specifics for CppCompileAction.
 */
public class FeatureConfigurationCommandLine extends CommandLine {

  private final FeatureConfiguration featureConfiguration;
  private final CcToolchainVariables buildVariables;
  private final String actionName;
  private final boolean usePic;

  /**
   * Create a {@link CommandLine} instance from given {@link FeatureConfiguration} and {@link
   * CcToolchainVariables}.
   */
  public static FeatureConfigurationCommandLine from(
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables ccToolchainVariables,
      String actionName) {
    return new FeatureConfigurationCommandLine(
        featureConfiguration, ccToolchainVariables, actionName, /* usePic= */ false);
  }

  /**
   * Do not use for new code.
   *
   * <p>TODO(hlopko): Refactor to remove the usePic field.
   *
   * @deprecated Only there for legacy/technical debt reasons for ThinLTO.
   */
  @Deprecated
  public static FeatureConfigurationCommandLine forLtoBackendAction(
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables ccToolchainVariables,
      boolean usePic) {
    return new FeatureConfigurationCommandLine(
        featureConfiguration, ccToolchainVariables, CppActionNames.LTO_BACKEND, usePic);
  }

  private FeatureConfigurationCommandLine(
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables ccToolchainVariables,
      String actionName,
      boolean usePic) {
    this.featureConfiguration = featureConfiguration;
    this.buildVariables = ccToolchainVariables;
    this.actionName = actionName;
    this.usePic = usePic;
  }
  @Override
  public Iterable<String> arguments() throws CommandLineExpansionException {
    return arguments(/* artifactExpander= */ null);
  }

  @Override
  public Iterable<String> arguments(ArtifactExpander artifactExpander)
      throws CommandLineExpansionException {
    ImmutableList.Builder<String> args = ImmutableList.builder();
    try {
      args.addAll(
          featureConfiguration.getCommandLine(actionName, buildVariables, artifactExpander));
    } catch (ExpansionException e) {
      throw new CommandLineExpansionException(e.getMessage());
    }
    // If this is a PIC compile (set based on the CppConfiguration), the PIC
    // option should be added after the rest of the command line so that it
    // cannot be overridden. This is consistent with the ordering in the
    // CppCompileAction's compiler options.
    if (usePic) {
      args.add("-fPIC");
    }
    return args.build();
  }
}
