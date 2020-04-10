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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.repository.RequestRepositoryInformationEvent;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Reports cycles of recursive import of Skylark files.
 */
public class SkylarkModuleCycleReporter implements CyclesReporter.SingleCycleReporter {

  private static final Predicate<SkyKey> IS_SKYLARK_MODULE_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.SKYLARK_IMPORTS_LOOKUP);

  private static final Predicate<SkyKey> IS_PACKAGE_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE);

  private static final Predicate<SkyKey> IS_PACKAGE_LOOKUP =
      SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE_LOOKUP);

  private static final Predicate<SkyKey> IS_WORKSPACE_FILE =
      SkyFunctions.isSkyFunction(WorkspaceFileValue.WORKSPACE_FILE);

  private static final Predicate<SkyKey> IS_REPOSITORY =
      SkyFunctions.isSkyFunction(SkyFunctions.REPOSITORY);

  private static final Predicate<SkyKey> IS_REPOSITORY_DIRECTORY =
      SkyFunctions.isSkyFunction(SkyFunctions.REPOSITORY_DIRECTORY);

  private static final Predicate<SkyKey> IS_SKYLARK_IMPORTS_LOOKUP =
      SkyFunctions.isSkyFunction(SkyFunctions.SKYLARK_IMPORTS_LOOKUP);

  private static final Predicate<SkyKey> IS_EXTERNAL_PACKAGE =
      SkyFunctions.isSkyFunction(SkyFunctions.EXTERNAL_PACKAGE);

  private static final Predicate<SkyKey> IS_LOCAL_REPOSITORY_LOOKUP =
      SkyFunctions.isSkyFunction(SkyFunctions.LOCAL_REPOSITORY_LOOKUP);

  private static void requestRepoDefinitions(
      ExtendedEventHandler eventHandler, Iterable<SkyKey> repos) {
    for (SkyKey repo : repos) {
      if (repo instanceof RepositoryValue.Key) {
        eventHandler.post(
            new RequestRepositoryInformationEvent(
                ((RepositoryValue.Key) repo).argument().strippedName()));
      }
    }
  }

  @Override
  public boolean maybeReportCycle(
      SkyKey topLevelKey,
      CycleInfo cycleInfo,
      boolean alreadyReported,
      ExtendedEventHandler eventHandler) {
    ImmutableList<SkyKey> pathToCycle = cycleInfo.getPathToCycle();
    ImmutableList<SkyKey> cycle = cycleInfo.getCycle();
    if (pathToCycle.isEmpty()) {
      return false;
    }
    SkyKey lastPathElement = pathToCycle.get(pathToCycle.size() - 1);
    if (alreadyReported) {
      return true;
    } else if (Iterables.all(cycle, IS_SKYLARK_MODULE_SKY_KEY)
        // The last element before the cycle has to be a PackageFunction, SkylarkModule, or the
        // WORKSPACE
        && (IS_PACKAGE_SKY_KEY.apply(lastPathElement)
            || IS_SKYLARK_MODULE_SKY_KEY.apply(lastPathElement)
            || IS_WORKSPACE_FILE.apply(lastPathElement))) {

      Function<SkyKey, String> printer =
          new Function<SkyKey, String>() {
            @Override
            public String apply(SkyKey input) {
              if (input.argument() instanceof StarlarkImportLookupValue.SkylarkImportLookupKey) {
                return ((StarlarkImportLookupValue.SkylarkImportLookupKey) input.argument())
                    .importLabel.toString();
              } else if (input.argument() instanceof PackageIdentifier) {
                return ((PackageIdentifier) input.argument()) + "/BUILD";
              } else if (input.argument() instanceof WorkspaceFileValue.WorkspaceFileKey) {
                return ((WorkspaceFileValue.WorkspaceFileKey) input.argument())
                    .getPath()
                    .getRootRelativePath()
                    .toString();
              } else {
                throw new UnsupportedOperationException();
              }
            }
          };

      StringBuilder cycleMessage =
          new StringBuilder().append("cycle detected in extension files: ");

      // go back the path that lead to the cycle till we found the BUILD or WORKSPACE
      // file that lead to the circular load.
      int startIndex = pathToCycle.size() - 1;
      while (startIndex > 0
          && (IS_PACKAGE_SKY_KEY.apply(pathToCycle.get(startIndex - 1))
              || IS_SKYLARK_MODULE_SKY_KEY.apply(pathToCycle.get(startIndex - 1))
              || IS_WORKSPACE_FILE.apply(pathToCycle.get(startIndex - 1)))) {
        startIndex--;
      }
      for (int i = startIndex; i < pathToCycle.size(); i++) {
        cycleMessage.append("\n    ").append(printer.apply(pathToCycle.get(i)));
      }
      AbstractLabelCycleReporter.printCycle(cycleInfo.getCycle(), cycleMessage, printer);
      // TODO(bazel-team): it would be nice to pass the Location of the load Statement in the
      // BUILD file.
      eventHandler.handle(Event.error(null, cycleMessage.toString()));
      return true;
    } else if (Iterables.all(
        cycle, Predicates.or(IS_PACKAGE_LOOKUP, IS_REPOSITORY, IS_REPOSITORY_DIRECTORY))) {
      StringBuilder cycleMessage =
          new StringBuilder().append("Circular definition of repositories:");
      Iterable<SkyKey> repos = Iterables.filter(cycle, IS_REPOSITORY);
      Function<SkyKey, String> printer =
          new Function<SkyKey, String>() {
            @Override
            public String apply(SkyKey input) {
              if (input instanceof RepositoryValue.Key) {
                return ((RepositoryValue.Key) input).argument().getName();
              } else {
                throw new UnsupportedOperationException();
              }
            }
          };
      AbstractLabelCycleReporter.printCycle(ImmutableList.copyOf(repos), cycleMessage, printer);
      eventHandler.handle(Event.error(null, cycleMessage.toString()));
      // To help debugging, request that the information be printed about where the respective
      // repositories were defined.
      requestRepoDefinitions(eventHandler, repos);
      return true;
    } else if (Iterables.any(cycle, IS_REPOSITORY) && Iterables.any(cycle, IS_WORKSPACE_FILE)) {
      Iterable<SkyKey> repos =
          Iterables.filter(Iterables.concat(pathToCycle, cycle), IS_REPOSITORY);

      StringBuilder message = new StringBuilder();

      if (Iterables.any(cycle, IS_SKYLARK_IMPORTS_LOOKUP)) {
        Label fileLabel =
            ((StarlarkImportLookupValue.SkylarkImportLookupKey)
                    Iterables.getLast(Iterables.filter(cycle, IS_SKYLARK_IMPORTS_LOOKUP)))
                .getImportLabel();
        message.append("Failed to load Starlark extension '").append(fileLabel).append("'.\n");
      }

      message
          .append("Cycle in the workspace file detected. ")
          .append("This indicates that a repository is used prior to being defined.\n")
          .append(
              "The following chain of repository dependencies lead to the missing definition.\n");
      for (SkyKey repo : repos) {
        if (repo instanceof RepositoryValue.Key) {
          message
              .append(" - ")
              .append(((RepositoryValue.Key) repo).argument().getName())
              .append("\n");
        }
      }
      SkyKey missingRepo = Iterables.getLast(repos);
      if (missingRepo instanceof RepositoryValue.Key) {
        message
            .append("This could either mean you have to add the '")
            .append(((RepositoryValue.Key) missingRepo).argument().getName())
            .append("' repository with a statement like `http_archive` in your WORKSPACE file")
            .append(" (note that transitive dependencies are not added automatically), or move")
            .append(" an existing definition earlier in your WORKSPACE file.");
      }
      eventHandler.handle(Event.error(message.toString()));
      // To help debugging, request that the information be printed about where the respective
      // repositories were defined.
      requestRepoDefinitions(eventHandler, repos);
      return true;
    } else if (Iterables.any(cycle, IS_SKYLARK_IMPORTS_LOOKUP)) {
      Label fileLabel =
          ((StarlarkImportLookupValue.SkylarkImportLookupKey)
                  Iterables.getLast(Iterables.filter(cycle, IS_SKYLARK_IMPORTS_LOOKUP)))
              .getImportLabel();
      eventHandler.handle(
          Event.error(null, "Failed to load Starlark extension '" + fileLabel + "'.\n"));
        return true;
    } else if (Iterables.any(cycle, IS_PACKAGE_LOOKUP)) {
      PackageIdentifier pkg =
          (PackageIdentifier)
              Iterables.getLast(Iterables.filter(cycle, IS_PACKAGE_LOOKUP)).argument();
      eventHandler.handle(Event.error(null, "cannot load package '" + pkg + "'"));
      return true;
    }
    return false;
  }
}
