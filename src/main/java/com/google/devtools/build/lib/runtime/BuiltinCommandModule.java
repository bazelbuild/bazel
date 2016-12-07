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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.CanonicalizeCommand;
import com.google.devtools.build.lib.runtime.commands.CleanCommand;
import com.google.devtools.build.lib.runtime.commands.CoverageCommand;
import com.google.devtools.build.lib.runtime.commands.DumpCommand;
import com.google.devtools.build.lib.runtime.commands.HelpCommand;
import com.google.devtools.build.lib.runtime.commands.InfoCommand;
import com.google.devtools.build.lib.runtime.commands.MobileInstallCommand;
import com.google.devtools.build.lib.runtime.commands.ProfileCommand;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.runtime.commands.RunCommand;
import com.google.devtools.build.lib.runtime.commands.ShutdownCommand;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.runtime.commands.VersionCommand;
import com.google.devtools.common.options.OptionsProvider;

/**
 * Internal module for the built-in commands.
 */
public final class BuiltinCommandModule extends BlazeModule {
  @Override
  public void serverInit(OptionsProvider startupOptions, ServerBuilder builder) {
    builder.addCommands(
        new BuildCommand(),
        new CanonicalizeCommand(),
        new CleanCommand(),
        new CoverageCommand(),
        new DumpCommand(),
        new HelpCommand(),
        new InfoCommand(),
        new MobileInstallCommand(),
        new ProfileCommand(),
        new QueryCommand(),
        new RunCommand(),
        new ShutdownCommand(),
        new TestCommand(),
        new VersionCommand());
  }
}
