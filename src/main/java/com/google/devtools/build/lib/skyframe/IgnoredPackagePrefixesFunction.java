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

import static com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction.VENDOR_DIRECTORY;

import com.google.common.collect.ImmutableList;
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
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
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
    } catch (InvalidPathException e) {
      throw new IgnoredPatternsFunctionException(
          new InvalidIgnorePathException(e, patternFile.asPath().toString()));
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey key, Environment env)
      throws IgnoredPatternsFunctionException, InterruptedException {
    RepositoryName repositoryName = (RepositoryName) key.argument();

    RepoFileValue repoFileValue = (RepoFileValue) env.getValue(RepoFileValue.key(repositoryName));
    if (env.valuesMissing()) {
      return null;
    }

    ImmutableSet.Builder<PathFragment> ignoredPackagePrefixesBuilder = ImmutableSet.builder();
    if (!ignoredPackagePrefixesFile.equals(PathFragment.EMPTY_FRAGMENT)) {
      PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
      if (env.valuesMissing()) {
        return null;
      }

      if (repositoryName.isMain()) {
        PathFragment vendorDir = null;
        if (VENDOR_DIRECTORY.get(env).isPresent()) {
          vendorDir = VENDOR_DIRECTORY.get(env).get().asFragment();
        }

        for (Root packagePathEntry : pkgLocator.getPathEntries()) {
          PathFragment workspaceRoot = packagePathEntry.asPath().asFragment();
          if (vendorDir != null && vendorDir.startsWith(workspaceRoot)) {
            ignoredPackagePrefixesBuilder.add(vendorDir.relativeTo(workspaceRoot));
          }

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

    return IgnoredPackagePrefixesValue.of(
        ignoredPackagePrefixesBuilder.build(),
        repoFileValue.ignoredDirectories());
  }

  private static final class PathFragmentLineProcessor
      implements LineProcessor<ImmutableSet<PathFragment>> {
    private final ImmutableSet.Builder<PathFragment> fragments = ImmutableSet.builder();

    @Override
    public boolean processLine(String line) {
      if (!line.isEmpty() && !line.startsWith("#")) {
        fragments.add(PathFragment.create(line));

        // This is called for its side-effects rather than its output.
        // Specifically, it validates that the line is a valid path. This
        // doesn't do much on UNIX machines where only NUL is an invalid
        // character but can reject paths on Windows.
        //
        // This logic would need to be adjusted if wildcards are ever supported
        // (https://github.com/bazelbuild/bazel/issues/7093).
        var unused = Path.of(line);
      }
      return true;
    }

    @Override
    public ImmutableSet<PathFragment> getResult() {
      return fragments.build();
    }
  }

  private static class InvalidIgnorePathException extends Exception {
    public InvalidIgnorePathException(InvalidPathException e, String path) {
      super("Invalid path in " + path + ": " + e);
    }
  }

  private static final class IgnoredPatternsFunctionException extends SkyFunctionException {
    public IgnoredPatternsFunctionException(InconsistentFilesystemException e) {
      super(e, Transience.TRANSIENT);
    }

    public IgnoredPatternsFunctionException(InvalidIgnorePathException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
