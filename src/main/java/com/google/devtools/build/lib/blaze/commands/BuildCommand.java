// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.blaze.commands;

import com.google.devtools.build.lib.blaze.BlazeCommand;
import com.google.devtools.build.lib.blaze.BlazeRuntime;
import com.google.devtools.build.lib.blaze.Command;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequest.BuildRequestOptions;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseRunner;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.view.BuildView;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

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
                     LoadingPhaseRunner.Options.class,
                     BuildConfiguration.Options.class,
                   },
         usesConfigurationOptions = true,
         shortDescription = "Builds the specified targets.",
         allowResidue = true,
         help = "resource:build.txt")
public final class BuildCommand implements BlazeCommand {

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser) { }

  @Override
  public ExitCode exec(BlazeRuntime runtime, OptionsProvider options, OutErr outErr) {
    BuildRequest request = BuildRequest.create(
        getClass().getAnnotation(Command.class).name(), options,
        runtime.getStartupOptionsProvider(),
        options.getResidue(),
        outErr, runtime.getCommandId(), runtime.getCommandStartTime());
    return runtime.getBuildTool().processRequest(request, null).getExitCondition();
  }
}
