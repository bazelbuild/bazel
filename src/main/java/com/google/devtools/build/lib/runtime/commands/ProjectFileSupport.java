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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.runtime.ProjectFile;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.util.List;

/**
 * Provides support for implementations for {@link BlazeCommand} to work with {@link ProjectFile}.
 */
public final class ProjectFileSupport {
  static final String PROJECT_FILE_PREFIX = "+";
  
  private ProjectFileSupport() {}

  /**
   * Reads any project files specified on the command line and updates the options parser
   * accordingly. If project files cannot be read or if they contain unparsable options, or if they
   * are not enabled, then it throws an exception instead.
   */
  public static void handleProjectFiles(
      ExtendedEventHandler eventHandler, ProjectFile.Provider projectFileProvider,
      Path workspaceDir, Path workingDir, OptionsParser optionsParser, String command)
          throws OptionsParsingException {
    List<String> targets = optionsParser.getResidue();
    if (projectFileProvider != null && !targets.isEmpty()
        && targets.get(0).startsWith(PROJECT_FILE_PREFIX)) {
      if (targets.size() > 1) {
        throw new OptionsParsingException("Cannot handle more than one +<file> argument yet");
      }
      if (!optionsParser.getOptions(CommonCommandOptions.class).allowProjectFiles) {
        throw new OptionsParsingException("project file support is not enabled. "
                                          + "Pass --experimental_allow_project_files to enable.");
      }
      // TODO(bazel-team): This is currently treated as a path relative to the workspace - if the
      // cwd is a subdirectory of the workspace, that will be surprising, and we should interpret it
      // relative to the cwd instead.
      PathFragment projectFilePath = PathFragment.create(targets.get(0).substring(1));
      List<Path> packagePath =
          PathPackageLocator.create(
                  // We only need a non-null outputBase for the PathPackageLocator if we support
                  // external
                  // repositories, which we don't for project files.
                  null,
                  optionsParser.getOptions(PackageCacheOptions.class).packagePath,
                  eventHandler,
                  workspaceDir,
                  workingDir,
                  BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY)
              .getPathEntries();
      ProjectFile projectFile = projectFileProvider.getProjectFile(
          workingDir, packagePath, projectFilePath);
      eventHandler.handle(Event.info("Using " + projectFile.getName()));

      optionsParser.parse(
          PriorityCategory.RC_FILE, projectFile.getName(), projectFile.getCommandLineFor(command));
      eventHandler.post(new GotProjectFileEvent(projectFile.getName()));
    }
  }

  /**
   * Returns a list of targets from the options residue. If a project file is supplied as the first
   * argument, it will be ignored, on the assumption that handleProjectFiles() has been called to
   * process it.
   */
  public static List<String> getTargets(
      ProjectFile.Provider projectFileProvider, OptionsProvider options) {
    List<String> targets = options.getResidue();
    if (projectFileProvider != null && !targets.isEmpty()
        && targets.get(0).startsWith(PROJECT_FILE_PREFIX)) {
      return targets.subList(1, targets.size());
    }
    return targets;
  }
}
