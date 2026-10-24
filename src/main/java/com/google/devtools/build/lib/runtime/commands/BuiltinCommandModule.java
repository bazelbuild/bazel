// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.query2.query.hash.DirectTargetHashFunction;
import com.google.devtools.build.lib.query2.query.hash.RegisteredToolchainsByTypeFunction;
import com.google.devtools.build.lib.query2.query.hash.RepoContentDigestFunction;
import com.google.devtools.build.lib.query2.query.hash.SourceFileDigestFunction;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.common.options.OptionsParsingResult;

/** Internal module for the built-in commands. */
public class BuiltinCommandModule extends BlazeModule {
  private final RunCommand runCommand;

  protected BuiltinCommandModule(RunCommand runCommand) {
    this.runCommand = runCommand;
  }

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
    builder.addCommands(
        new BuildCommand(),
        new CanonicalizeCommand(),
        new CleanCommand(),
        new CoverageCommand(),
        new DumpCommand(),
        new HelpCommand(),
        new InfoCommand(),
        new PrintActionCommand(),
        new QueryCommand(),
        runCommand,
        new ShutdownCommand(),
        new TestCommand(),
        new VersionCommand(),
        new AqueryCommand(),
        new CqueryCommand(),
        new ConfigCommand());
    // Only enable the "license" command when this binary has an embedded LICENSE file.
    if (LicenseCommand.isSupported()) {
      builder.addCommands(new LicenseCommand());
    }
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    // Used by query --generate_hashes.
    builder
        .addSkyFunction(DirectTargetHashFunction.NAME, new DirectTargetHashFunction(directories))
        .addSkyFunction(SourceFileDigestFunction.NAME, new SourceFileDigestFunction())
        .addSkyFunction(RepoContentDigestFunction.NAME, new RepoContentDigestFunction(directories))
        .addSkyFunction(
            RegisteredToolchainsByTypeFunction.NAME, new RegisteredToolchainsByTypeFunction());
  }
}
