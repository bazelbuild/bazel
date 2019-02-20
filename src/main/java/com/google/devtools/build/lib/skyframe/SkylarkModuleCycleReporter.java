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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
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

  private static final Predicate<SkyKey> IS_REPOSITORY_DIRECTORY =
      SkyFunctions.isSkyFunction(SkyFunctions.REPOSITORY_DIRECTORY);

  private static final Predicate<SkyKey> IS_AST_FILE_LOOKUP =
      SkyFunctions.isSkyFunction(SkyFunctions.AST_FILE_LOOKUP);

  private static final Predicate<SkyKey> IS_EXTERNAL_PACKAGE =
      SkyFunctions.isSkyFunction(SkyFunctions.EXTERNAL_PACKAGE);

  private static final Predicate<SkyKey> IS_LOCAL_REPOSITORY_LOOKUP =
      SkyFunctions.isSkyFunction(SkyFunctions.LOCAL_REPOSITORY_LOOKUP);

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
        // The last element before the cycle has to be a PackageFunction or SkylarkModule.
        && (IS_PACKAGE_SKY_KEY.apply(lastPathElement)
            || IS_SKYLARK_MODULE_SKY_KEY.apply(lastPathElement))) {

      Function printer =
          new Function<SkyKey, String>() {
            @Override
            public String apply(SkyKey input) {
              if (input.argument() instanceof SkylarkImportLookupValue.SkylarkImportLookupKey) {
                return ((SkylarkImportLookupValue.SkylarkImportLookupKey) input.argument())
                    .importLabel.toString();
              } else if (input.argument() instanceof PackageIdentifier) {
                return ((PackageIdentifier) input.argument()) + "/BUILD";
              } else {
                throw new UnsupportedOperationException();
              }
            }
          };

      StringBuilder cycleMessage =
          new StringBuilder()
              .append("cycle detected in extension files: ")
              .append("\n    ")
              .append(printer.apply(lastPathElement));

      AbstractLabelCycleReporter.printCycle(cycleInfo.getCycle(), cycleMessage, printer);
      // TODO(bazel-team): it would be nice to pass the Location of the load Statement in the
      // BUILD file.
      eventHandler.handle(Event.error(null, cycleMessage.toString()));
      return true;
    } else if (Iterables.any(cycle, IS_WORKSPACE_FILE)
        || IS_REPOSITORY_DIRECTORY.apply(lastPathElement)
        || IS_PACKAGE_SKY_KEY.apply(lastPathElement)
        || IS_EXTERNAL_PACKAGE.apply(lastPathElement)
        || IS_LOCAL_REPOSITORY_LOOKUP.apply(lastPathElement)) {
      // We have a cycle in the workspace file, report as such.
      if (Iterables.any(cycle, IS_AST_FILE_LOOKUP)) {
        Label fileLabel =
            (Label) Iterables.getLast(Iterables.filter(cycle, IS_AST_FILE_LOOKUP)).argument();
        String repositoryName = fileLabel.getPackageIdentifier().getRepository().strippedName();
        eventHandler.handle(
            Event.error(
                null,
                "Failed to load Starlark extension '"
                    + fileLabel
                    + "'.\n"
                    + "It usually happens when the repository is not defined prior to being used.\n"
                    + "This could either mean you have to add the '"
                    + fileLabel.getWorkspaceName()
                    + "' repository with a statement like `http_archive` in your WORKSPACE file"
                    + " (note that transitive dependencies are not added automatically), or"
                    + " the repository '"
                    + repositoryName
                    + "' was defined too late in your WORKSPACE file."));
        return true;
      } else if (Iterables.any(cycle, IS_PACKAGE_LOOKUP)) {
        eventHandler.handle(
            Event.error(null, "cycle detected loading "
                + String.join(
                    " ", lastPathElement.functionName().toString().toLowerCase().split("_"))
                + " '" + lastPathElement.argument().toString() + "'"));
        return true;
      }
    }
    return false;
  }
}
