// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.devtools.build.skyframe.SkyFunctionException.Transience.PERSISTENT;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/** A {@link SkyFunction} that loads metadata from a PROJECT.scl file. */
public class ProjectFunction implements SkyFunction {

  private static final String OWNED_CODE_PATHS_KEY = "owned_code_paths";

  // The set of top level reserved globals in the PROJECT.scl file.
  private static final ImmutableSet<String> RESERVED_GLOBALS =
      ImmutableSet.of(OWNED_CODE_PATHS_KEY);

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ProjectFunctionException, InterruptedException {
    ProjectValue.Key key = (ProjectValue.Key) skyKey.argument();

    BzlLoadValue bzlLoadValue;
    try {
      bzlLoadValue =
          (BzlLoadValue)
              env.getValueOrThrow(
                  BzlLoadValue.keyForBuild(key.getProjectFile()), BzlLoadFailedException.class);
    } catch (BzlLoadFailedException e) {
      throw new ProjectFunctionException(e, PERSISTENT);
    }
    if (bzlLoadValue == null) {
      return null;
    }

    Object ownedCodePathsRaw = bzlLoadValue.getModule().getGlobal(OWNED_CODE_PATHS_KEY);

    // Crude typechecking to prevent server crashes.
    @SuppressWarnings("unchecked")
    Collection<? extends String> ownedCodePaths =
        switch (ownedCodePathsRaw) {
          case null -> ImmutableSet.of();
          case Collection<?> xs -> {
            for (Object x : xs) {
              if (!(x instanceof String)) {
                throw new ProjectFunctionException(
                    new TypecheckFailureException(
                        "expected a list of strings, got element of " + x.getClass()));
              }
            }
            yield (Collection<String>) xs;
          }
          default ->
              throw new ProjectFunctionException(
                  new TypecheckFailureException(
                      "expected a list of strings, got " + ownedCodePathsRaw.getClass()));
        };

    ImmutableMap<String, Object> residualGlobals =
        bzlLoadValue.getModule().getGlobals().entrySet().stream()
            .filter(entry -> !RESERVED_GLOBALS.contains(entry.getKey()))
            .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));

    return new ProjectValue(ImmutableSet.copyOf(ownedCodePaths), residualGlobals);
  }

  private static final class TypecheckFailureException extends Exception {
    TypecheckFailureException(String msg) {
      super(msg);
    }
  }

  private static final class ProjectFunctionException extends SkyFunctionException {
    ProjectFunctionException(TypecheckFailureException cause) {
      super(cause, PERSISTENT);
    }

    ProjectFunctionException(BzlLoadFailedException e, Transience transience) {
      super(e, transience);
    }
  }
}
