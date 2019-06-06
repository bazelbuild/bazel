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
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;

/**
 * A representation of command line for {@link LtoBackendAction}.
 *
 * <p>We have this class so we're able to defer construction of the command line to execution phase.
 */
public class LtoBackendCommandLine extends CommandLine {

  private final FeatureConfiguration featureConfiguration;
  private final CcToolchainVariables buildVariables;
  private final boolean usePic;

  LtoBackendCommandLine(
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables ccToolchainVariables,
      boolean usePic) {
    this.featureConfiguration = featureConfiguration;
    this.buildVariables = ccToolchainVariables;
    this.usePic = usePic;
  }

  @Override
  public Iterable<String> arguments() throws CommandLineExpansionException {
    ImmutableList.Builder<String> args = ImmutableList.builder();
    try {
      args.addAll(featureConfiguration.getCommandLine(CppActionNames.LTO_BACKEND, buildVariables));
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
