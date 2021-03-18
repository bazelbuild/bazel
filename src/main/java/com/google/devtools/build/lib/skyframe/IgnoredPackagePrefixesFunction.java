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
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
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
 * A {@link SkyFunction} for {@link IgnoredPackagePrefixesValue}.
 *
 * <p>It is used to implement the `.bazelignore` feature.
 */
public class IgnoredPackagePrefixesFunction implements SkyFunction {
  private final PathFragment ignoredPackagePrefixesFile;

  public IgnoredPackagePrefixesFunction(PathFragment ignoredPackagePrefixesFile) {
    this.ignoredPackagePrefixesFile = ignoredPackagePrefixesFile;
  }

  public static void getIgnoredPackagePrefixes(
      RootedPath patternFile, ImmutableSet.Builder<PathFragment> ignoredPackagePrefixesBuilder)
      throws IgnoredPatternsFunctionException {
    try (InputStreamReader reader =
        new InputStreamReader(patternFile.asPath().getInputStream(), StandardCharsets.UTF_8)) {
      ignoredPackagePrefixesBuilder.addAll(
          CharStreams.readLines(reader, new PathFragmentLineProcessor()));
    } catch (IOException e) {
      String errorMessage = e.getMessage() != null ? "error '" + e.getMessage() + "'" : "an error";
      throw new IgnoredPatternsFunctionException(
          new InconsistentFilesystemException(
              patternFile.asPath()
                  + " is not readable because: "
                  + errorMessage
                  + ". Was it modified mid-build?"));
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey key, Environment env)
      throws SkyFunctionException, InterruptedException {
    RepositoryName repositoryName = (RepositoryName) key.argument();

    ImmutableSet.Builder<PathFragment> ignoredPackagePrefixesBuilder = ImmutableSet.builder();
    if (!ignoredPackagePrefixesFile.equals(PathFragment.EMPTY_FRAGMENT)) {
      PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
      if (env.valuesMissing()) {
        return null;
      }

      if (repositoryName.isMain()) {
        for (Root packagePathEntry : pkgLocator.getPathEntries()) {
          RootedPath rootedPatternFile =
              RootedPath.toRootedPath(packagePathEntry, ignoredPackagePrefixesFile);
          FileValue patternFileValue = (FileValue) env.getValue(FileValue.key(rootedPatternFile));
          if (patternFileValue == null) {
            return null;
          }
          if (patternFileValue.isFile()) {
            getIgnoredPackagePrefixes(rootedPatternFile, ignoredPackagePrefixesBuilder);
            break;
          }
        }
      } else {
        // Make sure the repository is fetched.
        RepositoryDirectoryValue repositoryValue =
            (RepositoryDirectoryValue) env.getValue(RepositoryDirectoryValue.key(repositoryName));
        if (repositoryValue == null) {
          return null;
        }
        if (repositoryValue.repositoryExists()) {
          RootedPath rootedPatternFile =
              RootedPath.toRootedPath(
                  Root.fromPath(repositoryValue.getPath()), ignoredPackagePrefixesFile);
          FileValue patternFileValue = (FileValue) env.getValue(FileValue.key(rootedPatternFile));
          if (patternFileValue == null) {
            return null;
          }
          if (patternFileValue.isFile()) {
            getIgnoredPackagePrefixes(rootedPatternFile, ignoredPackagePrefixesBuilder);
          }
        }
      }
    }

    return IgnoredPackagePrefixesValue.of(ignoredPackagePrefixesBuilder.build());
  }

  private static final class PathFragmentLineProcessor
      implements LineProcessor<ImmutableSet<PathFragment>> {
    private final ImmutableSet.Builder<PathFragment> fragments = ImmutableSet.builder();

    @Override
    public boolean processLine(String line) {
      if (!line.isEmpty() && !line.startsWith("#")) {
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

  private static final class IgnoredPatternsFunctionException extends SkyFunctionException {
    public IgnoredPatternsFunctionException(InconsistentFilesystemException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
