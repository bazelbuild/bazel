// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Optional;
import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.ErrorDeterminingRepositoryException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceFileHelper;
import com.google.devtools.build.lib.skyframe.PackageFunction.PackageFunctionException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import javax.annotation.Nullable;

/** SkyFunction for {@link LocalRepositoryLookupValue}s. */
public class LocalRepositoryLookupFunction implements SkyFunction {

  private final ExternalPackageHelper externalPackageHelper;

  public LocalRepositoryLookupFunction(ExternalPackageHelper externalPackageHelper) {
    this.externalPackageHelper = externalPackageHelper;
  }

  @Override
  @Nullable
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  // Implementation note: Although LocalRepositoryLookupValue.NOT_FOUND exists, it should never be
  // returned from this method.
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RootedPath directory = (RootedPath) skyKey.argument();

    // Is this the root directory? If so, we're in the MAIN repository. This assumes that the main
    // repository has a WORKSPACE in the root directory, but Bazel will have failed with an error
    // before this can be called if that is incorrect.
    if (directory.getRootRelativePath().equals(PathFragment.EMPTY_FRAGMENT)) {
      return LocalRepositoryLookupValue.mainRepository();
    }

    // Does this directory contain a WORKSPACE file?
    Optional<Boolean> maybeWorkspaceFileExists = maybeGetWorkspaceFileExistence(env, directory);
    if (!maybeWorkspaceFileExists.isPresent()) {
      return null;
    } else if (maybeWorkspaceFileExists.get()) {
      Optional<LocalRepositoryLookupValue> maybeRepository =
          maybeCheckWorkspaceForRepository(env, directory);
      if (!maybeRepository.isPresent()) {
        return null;
      }
      LocalRepositoryLookupValue repository = maybeRepository.get();
      // If the repository that was discovered doesn't exist, continue recursing.
      if (repository.exists()) {
        return repository;
      }
    }

    // If we haven't found a repository yet, check the parent directory.
    return env.getValue(LocalRepositoryLookupValue.key(directory.getParentDirectory()));
  }

  private Optional<Boolean> maybeGetWorkspaceFileExistence(Environment env, RootedPath directory)
      throws InterruptedException, LocalRepositoryLookupFunctionException {
    try {
      RootedPath workspaceRootedFile = WorkspaceFileHelper.getWorkspaceRootedFile(directory, env);
      if (workspaceRootedFile == null) {
        return Optional.absent();
      }
      FileValue workspaceFileValue =
          (FileValue) env.getValueOrThrow(FileValue.key(workspaceRootedFile), IOException.class);
      if (workspaceFileValue == null) {
        return Optional.absent();
      }
      if (workspaceFileValue.isDirectory()) {
        // There is a directory named WORKSPACE, ignore it for checking repository existence.
        return Optional.of(false);
      }
      return Optional.of(workspaceFileValue.exists());
    } catch (InconsistentFilesystemException e) {
      throw new LocalRepositoryLookupFunctionException(
          new ErrorDeterminingRepositoryException(
              "InconsistentFilesystemException while checking if there is a WORKSPACE file in "
                  + directory.asPath().getPathString(),
              e),
          Transience.PERSISTENT);
    } catch (FileSymlinkException e) {
      throw new LocalRepositoryLookupFunctionException(
          new ErrorDeterminingRepositoryException(
              "FileSymlinkException while checking if there is a WORKSPACE file in "
                  + directory.asPath().getPathString(),
              e),
          Transience.PERSISTENT);
    } catch (IOException e) {
      throw new LocalRepositoryLookupFunctionException(
          new ErrorDeterminingRepositoryException(
              "IOException while checking if there is a WORKSPACE file in "
                  + directory.asPath().getPathString(),
              e),
          Transience.PERSISTENT);
    }
  }

  /**
   * Checks whether the directory exists and is a workspace root. Returns {@link Optional#absent()}
   * if Skyframe needs to re-run, {@link Optional#of(LocalRepositoryLookupValue)} otherwise.
   */
  private Optional<LocalRepositoryLookupValue> maybeCheckWorkspaceForRepository(
      Environment env, final RootedPath directory)
      throws InterruptedException, LocalRepositoryLookupFunctionException {
    RootedPath workspacePath = externalPackageHelper.findWorkspaceFile(env);
    if (env.valuesMissing()) {
      return Optional.absent();
    }

    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    do {
      WorkspaceFileValue value;
      try {
        value =
            (WorkspaceFileValue)
                env.getValueOrThrow(
                    workspaceKey, PackageFunctionException.class, NameConflictException.class);
        if (value == null) {
          return Optional.absent();
        }
      } catch (PackageFunctionException e) {
        // TODO(jcater): When WFF is rewritten to not throw a PFE, update this.
        throw new LocalRepositoryLookupFunctionException(
            new ErrorDeterminingRepositoryException(
                "PackageFunctionException while loading the root WORKSPACE file", e),
            Transience.PERSISTENT);
      } catch (NameConflictException e) {
        throw new LocalRepositoryLookupFunctionException(
            new ErrorDeterminingRepositoryException(
                "NameConflictException while loading the root WORKSPACE file", e),
            Transience.PERSISTENT);
      }

      Package externalPackage = value.getPackage();
      // Find all local_repository rules in the WORKSPACE, and check if any have a "path" attribute
      // the same as the requested directory.
      Iterable<Rule> localRepositories =
          Iterables.filter(
              externalPackage.getTargets(Rule.class),
              rule -> LocalRepositoryRule.NAME.equals(rule.getRuleClass()));
      Rule rule =
          Iterables.find(
              localRepositories,
              new Predicate<Rule>() {
                @Override
                public boolean apply(@Nullable Rule rule) {
                  AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
                  // Construct the path. If not absolute, it will be relative to the workspace.
                  Path localPath =
                      workspacePath.getRoot().getRelative(mapper.get("path", Type.STRING));
                  return directory.asPath().equals(localPath);
                }
              },
              null);
      if (rule != null) {
        try {
          String path = (String) rule.getAttr("path");
          return Optional.of(
              LocalRepositoryLookupValue.success(
                  RepositoryName.create("@" + rule.getName()), PathFragment.create(path)));
        } catch (LabelSyntaxException e) {
          // This shouldn't be possible if the rule name is valid, and it should already have been
          // validated.
          throw new LocalRepositoryLookupFunctionException(
              new ErrorDeterminingRepositoryException(
                  "LabelSyntaxException while creating the repository name from the rule "
                      + rule.getName(),
                  e),
              Transience.PERSISTENT);
        }
      }
      workspaceKey = value.next();

      // TODO(bazel-team): This loop can be quadratic in the number of load() statements, consider
      // rewriting or unrolling.
    } while (workspaceKey != null);

    return Optional.of(LocalRepositoryLookupValue.notFound());
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * LocalRepositoryLookupFunction#compute}.
   */
  private static final class LocalRepositoryLookupFunctionException extends SkyFunctionException {
    public LocalRepositoryLookupFunctionException(
        ErrorDeterminingRepositoryException e, Transience transience) {
      super(e, transience);
    }
  }
}
