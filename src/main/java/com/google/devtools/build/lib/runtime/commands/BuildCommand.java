// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequest.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.util.List;

/**
 * Handles the 'build' command on the Blaze command line, including targets
 * named by arguments passed to Blaze.
 */
@Command(name = "build",
         builds = true,
         options = { BuildRequestOptions.class,
                     ExecutionOptions.class,
                     PackageCacheOptions.class,
                     BuildView.Options.class,
                     LoadingOptions.class,
                   },
         usesConfigurationOptions = true,
         shortDescription = "Builds the specified targets.",
         allowResidue = true,
         completion = "label",
         help = "resource:build.txt")
public final class BuildCommand implements BlazeCommand {

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser)
      throws AbruptExitException {
    ProjectFileSupport.handleProjectFiles(env, optionsParser, "build");
  }

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    BlazeRuntime runtime = env.getRuntime();
    List<String> targets = ProjectFileSupport.getTargets(runtime, options);

    BuildRequest request = BuildRequest.create(
        getClass().getAnnotation(Command.class).name(), options,
        runtime.getStartupOptionsProvider(),
        targets,
        env.getReporter().getOutErr(), env.getCommandId(), env.getCommandStartTime());
    return new BuildTool(env).processRequest(request, null).getExitCondition();
  }
}
