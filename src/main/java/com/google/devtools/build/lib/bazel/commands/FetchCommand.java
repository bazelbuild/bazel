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
package com.google.devtools.build.lib.bazel.commands;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

/**
 * Fetches external repositories. Which is so fetch.
 */
@Command(name = "fetch",
    options = { PackageCacheOptions.class },
    help = "resource:fetch.txt",
    shortDescription = "Fetches external repositories that are prerequisites to the targets.",
    allowResidue = true,
    completion = "label")
public final class FetchCommand implements BlazeCommand {
  // TODO(kchodorow): add an option to force-fetch targets, even if they're already downloaded.
  // TODO(kchodorow): this would be a great time to check for difference and invalidate the upward
  //                  transitive closure for local repositories.
  // TODO(kchodorow): prevent fetching from being done during a build.

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser) { }

  @Override
  public ExitCode exec(BlazeRuntime runtime, OptionsProvider options) {
    if (options.getResidue().isEmpty()) {
      runtime.getReporter().handle(Event.error(String.format(
          "missing fetch expression. Type '%s help fetch' for syntax and help",
          Constants.PRODUCT_NAME)));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    try {
      runtime.setupPackageCache(
          options.getOptions(PackageCacheOptions.class),
          runtime.getDefaultsPackageContent());
    } catch (InterruptedException e) {
      runtime.getReporter().handle(Event.error("fetch interrupted"));
      return ExitCode.INTERRUPTED;
    } catch (AbruptExitException e) {
      runtime.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return e.getExitCode();
    }

    // Querying for all of the dependencies of the targets has the side-effect of populating the
    // Skyframe graph for external targets, which requires downloading them.
    String query = Joiner.on(" union ").join(options.getResidue());
    query = "deps(" + query + ")";

    AbstractBlazeQueryEnvironment<Target> env = QueryCommand.newQueryEnvironment(
        runtime,
        true,
        false,
        Lists.<String>newArrayList(), 4,
        Sets.<Setting>newHashSet());

    // 1. Parse query:
    QueryExpression expr;
    try {
      expr = QueryExpression.parse(query, env);
    } catch (QueryException e) {
      runtime.getReporter().handle(Event.error(
          null, "Error while parsing '" + query + "': " + e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    // 2. Evaluate expression:
    try {
      env.evaluateQuery(expr);
    } catch (QueryException e) {
      // Keep consistent with reportBuildFileError()
      runtime.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    return ExitCode.SUCCESS;
  }


}
