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
import com.google.devtools.build.lib.blaze.CommonCommandOptions;
import com.google.devtools.build.lib.blaze.ProjectFile;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequest.BuildRequestOptions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseRunner;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.BuildView;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
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
                     LoadingPhaseRunner.Options.class,
                     BuildConfiguration.Options.class,
                   },
         usesConfigurationOptions = true,
         shortDescription = "Builds the specified targets.",
         allowResidue = true,
         help = "resource:build.txt")
public final class BuildCommand implements BlazeCommand {
  private static final String PROJECT_FILE_PREFIX = "+";

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser)
      throws AbruptExitException {
    handleProjectFiles(runtime, optionsParser);
  }

  @Override
  public ExitCode exec(BlazeRuntime runtime, OptionsProvider options) {
    List<String> targets = options.getResidue();
    if (runtime.getProjectFileProvider(options) != null && targets.size() > 0
        && targets.get(0).startsWith(PROJECT_FILE_PREFIX)) {
      targets = targets.subList(1, targets.size());
    }

    BuildRequest request = BuildRequest.create(
        getClass().getAnnotation(Command.class).name(), options,
        runtime.getStartupOptionsProvider(),
        targets,
        runtime.getReporter().getOutErr(), runtime.getCommandId(), runtime.getCommandStartTime());
    return runtime.getBuildTool().processRequest(request, null).getExitCondition();
  }

  public static boolean allowProjectFiles(OptionsProvider options) {
    return options.getOptions(CommonCommandOptions.class).allowProjectFiles;
  }

  /**
   * Reads any project files specified on the command line and updates the options parser
   * accordingly. If project files cannot be read or if they contain unparsable options, then it
   * throws an exception instead.
   */
  public static void handleProjectFiles(BlazeRuntime runtime, OptionsParser optionsParser)
      throws AbruptExitException {
    List<String> targets = optionsParser.getResidue();
    ProjectFile.Provider projectFileProvider = runtime.getProjectFileProvider(optionsParser);
    if (projectFileProvider != null && targets.size() > 0
        && targets.get(0).startsWith(PROJECT_FILE_PREFIX)) {
      if (targets.size() > 1) {
        throw new AbruptExitException("Cannot handle more than one +<file> argument yet",
            ExitCode.COMMAND_LINE_ERROR);
      }
      // TODO(bazel-team): This is currently treated as a path relative to the workspace - if the
      // cwd is a subdirectory of the workspace, that will be surprising, and we should interpret it
      // relative to the cwd instead.
      PathFragment projectFilePath = new PathFragment(targets.get(0).substring(1));
      List<Path> packagePath = PathPackageLocator.create(
          optionsParser.getOptions(PackageCacheOptions.class).packagePath,
          runtime.getReporter(), runtime.getWorkspace(), runtime.getWorkingDirectory())
          .getPathEntries();
      ProjectFile projectFile = projectFileProvider.getProjectFile(
          packagePath, projectFilePath, "build");
      runtime.getReporter().handle(Event.info("Using " + projectFile.getName()));

      try {
        optionsParser.parse(
            OptionPriority.RC_FILE, projectFile.getName(), projectFile.getCommandLine());
      } catch (OptionsParsingException e) {
        throw new AbruptExitException(e.getMessage(), ExitCode.COMMAND_LINE_ERROR);
      }
    }
  }
}
