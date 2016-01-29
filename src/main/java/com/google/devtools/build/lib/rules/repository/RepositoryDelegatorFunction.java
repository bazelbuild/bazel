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

package com.google.devtools.build.lib.rules.repository;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A {@link SkyFunction} that implements delegation to the correct repository fetcher.
 *
 * <p>Each repository in the WORKSPACE file is represented by a {@link SkyValue} that is computed
 * by this function.
 */
public class RepositoryDelegatorFunction implements SkyFunction {

  // Mapping of rule class name to RepositoryFunction.
  private final ImmutableMap<String, RepositoryFunction> handlers;

  // This is a reference to isFetch in BazelRepositoryModule, which tracks whether the current
  // command is a fetch. Remote repository lookups are only allowed during fetches.
  private final AtomicBoolean isFetch;
  private final BlazeDirectories directories;

  public RepositoryDelegatorFunction(
      BlazeDirectories directories, ImmutableMap<String, RepositoryFunction> handlers,
      AtomicBoolean isFetch) {
    this.directories = directories;
    this.handlers = handlers;
    this.isFetch = isFetch;
  }

  private void setupRepositoryRoot(Path repoRoot) throws RepositoryFunctionException {
    try {
      FileSystemUtils.deleteTree(repoRoot);
      FileSystemUtils.createDirectoryAndParents(repoRoot.getParentDirectory());
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction
        .getRule(repositoryName, null, env);
    if (rule == null) {
      return null;
    }

    RepositoryFunction handler = handlers.get(rule.getRuleClass());
    if (handler == null) {
      throw new RepositoryFunctionException(new EvalException(
          Location.fromFile(directories.getWorkspace().getRelative("WORKSPACE")),
          "Could not find handler for " + rule), Transience.PERSISTENT);
    }

    Path repoRoot =
        RepositoryFunction.getExternalRepositoryDirectory(directories).getRelative(rule.getName());

    if (handler.isLocal()) {
      // Local repositories are always fetched because the operation is generally fast and they do
      // not depend on non-local data, so it does not make much sense to try to catch from across
      // server instances.
      setupRepositoryRoot(repoRoot);
      return handler.fetch(rule, repoRoot, env);
    }

    // We check the repository root for existence here, but we can't depend on the FileValue,
    // because it's possible that we eventually create that directory in which case the FileValue
    // and the state of the file system would be inconsistent.

    byte[] ruleSpecificData = handler.getRuleSpecificMarkerData(rule, env);
    if (ruleSpecificData == null) {
      return null;
    }
    boolean markerUpToDate = handler.isFilesystemUpToDate(rule, ruleSpecificData);
    if (markerUpToDate && repoRoot.exists()) {
      // Now that we know that it exists, we can declare a Skyframe dependency on the repository
      // root.
      FileValue repoRootValue = RepositoryFunction.getRepositoryDirectory(repoRoot, env);
      if (env.valuesMissing()) {
        return null;
      }

      // NB: This returns the wrong repository value for non-local new_* repository functions.
      // This should sort itself out automatically once the ExternalFilesHelper refactoring is
      // finally submitted.
      return RepositoryDirectoryValue.create(repoRootValue.realRootedPath().asPath());
    }

    if (isFetch.get()) {
      // Fetching enabled, go ahead.
      setupRepositoryRoot(repoRoot);
      SkyValue result = handler.fetch(rule, repoRoot, env);
      if (env.valuesMissing()) {
        return null;
      }

      // No new Skyframe dependencies must be added between calling the repository implementation
      // and writing the marker file because if they aren't computed, it would cause a Skyframe
      // restart thus calling the possibly very slow (networking, decompression...) fetch()
      // operation again. So we write the marker file here immediately.
      handler.writeMarkerFile(rule, ruleSpecificData);
      return result;
    }

    if (!repoRoot.exists()) {
      // The repository isn't on the file system, there is nothing we can do.
      throw new RepositoryFunctionException(new IOException(
          "to fix, run\n\tbazel fetch //...\nExternal repository " + repositoryName
              + " not found and fetching repositories is disabled."),
          Transience.TRANSIENT);
    }

    // Declare a Skyframe dependency so that this is re-evaluated when something happens to the
    // directory.
    FileValue repoRootValue = RepositoryFunction.getRepositoryDirectory(repoRoot, env);
    if (env.valuesMissing()) {
      return null;
    }

    // Try to build with whatever is on the file system and emit a warning.
    env.getListener().handle(Event.warn(rule.getLocation(), String.format(
        "External repository '%s' is not up-to-date and fetching is disabled. To update, "
        + "run the build without the '--nofetch' command line option.",
        rule.getName())));

    return RepositoryDirectoryValue.fetchingDelayed(repoRootValue.realRootedPath().asPath());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
