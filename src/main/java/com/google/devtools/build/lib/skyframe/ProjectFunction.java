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
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;

/** A {@link SkyFunction} that loads metadata from a PROJECT.scl file. */
public class ProjectFunction implements SkyFunction {

  /** The top level reserved globals in the PROJECT.scl file. */
  private enum ReservedGlobals {
    OWNED_CODE_PATHS("owned_code_paths"),

    ACTIVE_DIRECTORIES("active_directories");

    private final String key;

    ReservedGlobals(String key) {
      this.key = key;
    }

    String getKey() {
      return key;
    }
  }

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

    Object activeDirectoriesRaw =
        bzlLoadValue.getModule().getGlobal(ReservedGlobals.ACTIVE_DIRECTORIES.getKey());

    // Crude typechecking to prevent server crashes.
    // TODO: all of these typechecking should probably be handled by a proto spec.
    @SuppressWarnings("unchecked")
    ImmutableMap<String, Collection<String>> activeDirectories =
        switch (activeDirectoriesRaw) {
          case null -> ImmutableMap.of();
          case Dict<?, ?> dict -> {
            ImmutableMap.Builder<String, Collection<String>> builder = ImmutableMap.builder();
            for (Entry<?, ?> entry : dict.entrySet()) {
              Object k = entry.getKey();

              if (!(k instanceof String activeDirectoriesKey)) {
                throw new ProjectFunctionException(
                    new TypecheckFailureException(
                        "expected string, got element of " + k.getClass()));
              }

              Object values = entry.getValue();
              if (!(values instanceof Collection<?> activeDirectoriesValues)) {
                throw new ProjectFunctionException(
                    new TypecheckFailureException(
                        "expected list, got element of " + values.getClass()));
              }

              for (Object activeDirectory : activeDirectoriesValues) {
                if (!(activeDirectory instanceof String)) {
                  throw new ProjectFunctionException(
                      new TypecheckFailureException(
                          "expected a list of strings, got element of "
                              + activeDirectory.getClass()));
                }
              }

              builder.put(activeDirectoriesKey, (Collection<String>) values);
            }

            yield builder.buildOrThrow();
          }
          default ->
              throw new ProjectFunctionException(
                  new TypecheckFailureException(
                      "expected a map of string to list of strings, got "
                          + activeDirectoriesRaw.getClass()));
        };

    ImmutableMap<String, Object> residualGlobals =
        bzlLoadValue.getModule().getGlobals().entrySet().stream()
            .filter(
                entry ->
                    Arrays.stream(ReservedGlobals.values())
                        .noneMatch(global -> entry.getKey().equals(global.getKey())))
            .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));

    if (!activeDirectories.isEmpty() && activeDirectories.get("default") == null) {
      throw new ProjectFunctionException(
          new ActiveDirectoriesException(
              "non-empty active_directories must contain the 'default' key"));
    }

    return new ProjectValue(activeDirectories, residualGlobals);
  }

  private static final class TypecheckFailureException extends Exception {
    TypecheckFailureException(String msg) {
      super(msg);
    }
  }

  private static final class ActiveDirectoriesException extends Exception {
    ActiveDirectoriesException(String msg) {
      super(msg);
    }
  }

  private static final class ProjectFunctionException extends SkyFunctionException {
    ProjectFunctionException(TypecheckFailureException cause) {
      super(cause, PERSISTENT);
    }

    ProjectFunctionException(ActiveDirectoriesException cause) {
      super(cause, PERSISTENT);
    }

    ProjectFunctionException(BzlLoadFailedException e, Transience transience) {
      super(e, transience);
    }
  }
}
