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
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Implements delegation to the correct repository fetcher.
 */
public class RepositoryDelegatorFunction implements SkyFunction {

  // Mapping of rule class name to SkyFunction.
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

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction
        .getRule(repositoryName, null, env);
    if (rule == null) {
      return null;
    }

    // If Bazel isn't running a fetch command, we shouldn't be able to download anything. To
    // prevent having to rerun fetch on server restart, we check if the external repository
    // directory already exists and, if it does, just use that.
    if (!isFetch.get()) {
      FileValue repoRoot = RepositoryFunction.getRepositoryDirectory(
          RepositoryFunction
              .getExternalRepositoryDirectory(directories)
              .getRelative(rule.getName()), env);
      if (repoRoot == null) {
        return null;
      }
      Path repoPath = repoRoot.realRootedPath().asPath();
      if (!repoPath.exists()) {
        throw new RepositoryFunctionException(new IOException(
            "to fix, run\n\tbazel fetch //...\nExternal repository " + repositoryName
                + " not found"),
            Transience.TRANSIENT);
      }
      return RepositoryValue.create(repoPath);
    }

    RepositoryFunction handler = handlers.get(rule.getRuleClass());
    if (handler == null) {
      throw new RepositoryFunctionException(new EvalException(
          Location.fromFile(directories.getWorkspace().getRelative("WORKSPACE")),
          "Could not find handler for " + rule), Transience.PERSISTENT);
    }
    SkyKey key = new SkyKey(handler.getSkyFunctionName(), repositoryName);

    try {
      return env.getValueOrThrow(
          key, NoSuchPackageException.class, IOException.class, EvalException.class);
    } catch (NoSuchPackageException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
