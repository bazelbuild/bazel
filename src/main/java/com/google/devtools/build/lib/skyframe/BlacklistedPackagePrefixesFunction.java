// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.common.io.CharStreams;
import com.google.common.io.LineProcessor;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import javax.annotation.Nullable;

/**
 * A function that retrieves a set of blacklisted package pattern prefixes from the file given by
 * PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.
 */
public class BlacklistedPackagePrefixesFunction implements SkyFunction {
  @Nullable
  @Override
  public SkyValue compute(SkyKey key, Environment env)
      throws SkyFunctionException, InterruptedException {
    PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    PathFragment patternsFile = PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    if (patternsFile.equals(PathFragment.EMPTY_FRAGMENT)) {
      return new BlacklistedPackagePrefixesValue(ImmutableSet.<PathFragment>of());
    }

    for (Path packagePathEntry : pkgLocator.getPathEntries()) {
      RootedPath rootedPatternFile = RootedPath.toRootedPath(packagePathEntry, patternsFile);
      FileValue patternFileValue = (FileValue) env.getValue(FileValue.key(rootedPatternFile));
      if (patternFileValue == null) {
        return null;
      }
      if (patternFileValue.isFile()) {
        try {
          try (InputStreamReader reader =
              new InputStreamReader(rootedPatternFile.asPath().getInputStream(),
                  StandardCharsets.UTF_8)) {
            return new BlacklistedPackagePrefixesValue(
                CharStreams.readLines(reader, new PathFragmentLineProcessor()));
          }
        } catch (IOException e) {
          String errorMessage = e.getMessage() != null
              ? "error '" + e.getMessage() + "'" : "an error";
          throw new BlacklistedPatternsFunctionException(
              new InconsistentFilesystemException(
                  rootedPatternFile.asPath() + " is not readable because: " +  errorMessage
                      + ". Was it modified mid-build?"));
        }
      }
    }

    return new BlacklistedPackagePrefixesValue(ImmutableSet.<PathFragment>of());
  }

  private static final class PathFragmentLineProcessor
      implements LineProcessor<ImmutableSet<PathFragment>> {
    private final ImmutableSet.Builder<PathFragment> fragments = ImmutableSet.builder();

    @Override
    public boolean processLine(String line) throws IOException {
      if (!line.isEmpty()) {
        fragments.add(PathFragment.create(line));
      }
      return true;
    }

    @Override
    public ImmutableSet<PathFragment> getResult() {
      return fragments.build();
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class BlacklistedPatternsFunctionException extends SkyFunctionException {
    public BlacklistedPatternsFunctionException(InconsistentFilesystemException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
