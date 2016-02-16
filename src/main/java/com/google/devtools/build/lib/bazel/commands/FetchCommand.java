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
package com.google.devtools.build.lib.bazel.commands;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.rules.java.JavaOptions;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

/**
 * Fetches external repositories. Which is so fetch.
 */
@Command(name = FetchCommand.NAME,
    options = {
        PackageCacheOptions.class,
        FetchOptions.class,
        JavaOptions.class,
    },
    help = "resource:fetch.txt",
    shortDescription = "Fetches external repositories that are prerequisites to the targets.",
    allowResidue = true,
    completion = "label")
public final class FetchCommand implements BlazeCommand {
  // TODO(kchodorow): add an option to force-fetch targets, even if they're already downloaded.
  // TODO(kchodorow): this would be a great time to check for difference and invalidate the upward
  //                  transitive closure for local repositories.

  public static final String NAME = "fetch";

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser) { }

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    BlazeRuntime runtime = env.getRuntime();
    if (options.getResidue().isEmpty()) {
      env.getReporter().handle(Event.error(String.format(
          "missing fetch expression. Type '%s help fetch' for syntax and help",
          Constants.PRODUCT_NAME)));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    try {
      env.setupPackageCache(
          options.getOptions(PackageCacheOptions.class),
          runtime.getDefaultsPackageContent());
    } catch (InterruptedException e) {
      env.getReporter().handle(Event.error("fetch interrupted"));
      return ExitCode.INTERRUPTED;
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return e.getExitCode();
    }

    PackageCacheOptions pkgOptions = options.getOptions(PackageCacheOptions.class);
    if (!pkgOptions.fetch) {
      env.getReporter().handle(Event.error(null, "You cannot run fetch with --fetch=false"));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    // Querying for all of the dependencies of the targets has the side-effect of populating the
    // Skyframe graph for external targets, which requires downloading them. The JDK is required to
    // build everything but isn't counted as a dep in the build graph so we add it manually.
    JavaOptions javaOptions = options.getOptions(JavaOptions.class);
    ImmutableList.Builder<String> labelsToLoad = new ImmutableList.Builder<String>()
        .addAll(options.getResidue());
    if (String.valueOf(javaOptions.javaLangtoolsJar).equals(
        runtime.getRuleClassProvider().getToolsRepository() + JavaOptions.DEFAULT_LANGTOOLS)) {
      labelsToLoad.add(javaOptions.javaBase);
    } else {
      // TODO(kchodroow): Remove this when OS X isn't as hacky about finding the JVM. Our test
      // framework currently doesn't set up the JDK normally on OS X, so attempting to fetch
      // tools/jdk:jdk will cause errors.
      labelsToLoad.add(String.valueOf(javaOptions.javaToolchain));
    }
    String query = Joiner.on(" union ").join(labelsToLoad.build());
    query = "deps(" + query + ")";

    AbstractBlazeQueryEnvironment<Target> queryEnv = QueryCommand.newQueryEnvironment(
        env, options.getOptions(FetchOptions.class).keepGoing, false,
        Lists.<String>newArrayList(), 200, Sets.<Setting>newHashSet());

    // 1. Parse query:
    QueryExpression expr;
    try {
      expr = QueryExpression.parse(query, queryEnv);
    } catch (QueryException e) {
      env.getReporter().handle(Event.error(
          null, "Error while parsing '" + query + "': " + e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    // 2. Evaluate expression:
    try {
      queryEnv.evaluateQuery(expr, new Callback<Target>() {
        @Override
        public void process(Iterable<Target> partialResult)
            throws QueryException, InterruptedException {
          // Throw away the result.
        }
      });
    } catch (QueryException | InterruptedException e) {
      // Keep consistent with reportBuildFileError()
      env.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    env.getReporter().handle(
        Event.progress("All external dependencies fetched successfully."));
    return ExitCode.SUCCESS;
  }
}
