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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * Implements delegation to the correct repository fetcher.
 */
public class RepositoryDelegatorFunction implements SkyFunction {

  // Mapping of rule class name to SkyFunction.
  private final ImmutableMap<String, RepositoryFunction> handlers;

  public RepositoryDelegatorFunction(
      ImmutableMap<String, RepositoryFunction> handlers) {
    this.handlers = handlers;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, null, env);
    if (rule == null) {
      return null;
    }
    RepositoryFunction handler = handlers.get(rule.getRuleClass());
    if (handler == null) {
      throw new IllegalStateException("Could not find handler for " + rule);
    }
    SkyKey key = new SkyKey(handler.getSkyFunctionName(), repositoryName);

    try {
      return env.getValueOrThrow(
          key, NoSuchPackageException.class, IOException.class, EvalException.class);
    } catch (NoSuchPackageException e) {
      throw new RepositoryFunction.RepositoryFunctionException(e, Transience.PERSISTENT);
    } catch (IOException e) {
      throw new RepositoryFunction.RepositoryFunctionException(e, Transience.PERSISTENT);
    } catch (EvalException e) {
      throw new RepositoryFunction.RepositoryFunctionException(e, Transience.PERSISTENT);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
