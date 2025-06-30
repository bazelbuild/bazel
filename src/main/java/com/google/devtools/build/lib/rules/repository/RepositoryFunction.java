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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.AlreadyReportedException;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * Implementation of fetching various external repository types.
 *
 * <p>These objects are called from {@link RepositoryDelegatorFunction}.
 *
 * <p>External repositories come in two flavors: local and non-local.
 *
 * <p>Local ones are those whose fetching does not require access to any external resources (e.g.
 * network). These are always re-fetched on Bazel server restarts. This operation is fast (usually
 * just a few symlinks and maybe writing a BUILD file). {@code --nofetch} does not apply to local
 * repositories.
 *
 * <p>The up-to-dateness of non-local repositories is checked using a marker file under the output
 * base. When such a repository is fetched, data from the rule in the WORKSPACE file is written to
 * the marker file which is consulted on next server startup. If the rule hasn't changed, the
 * repository is not re-fetched.
 *
 * <p>Fetching repositories can be disabled using the {@code --nofetch} command line option. If a
 * repository is on the file system, Bazel just tries to use it and hopes for the best. If the
 * repository has never been fetched, Bazel errors out for lack of a better option. This is
 * implemented using {@link
 * com.google.devtools.build.lib.bazel.BazelRepositoryModule#REPOSITORY_VALUE_CHECKER} and a flag in
 * {@link RepositoryDirectoryValue} that tells Bazel whether the value in Skyframe is stale
 * according to the value of {@code --nofetch} or not.
 *
 * <p>When a rule in the WORKSPACE file is changed, the corresponding {@link
 * RepositoryDirectoryValue} is invalidated using the usual Skyframe route.
 */
public abstract class RepositoryFunction {

  /**
   * Exception thrown when something goes wrong accessing a remote repository.
   *
   * <p>This exception should be used by child classes to limit the types of exceptions {@link
   * RepositoryDelegatorFunction} has to know how to catch.
   */
  public static class RepositoryFunctionException extends SkyFunctionException {

    /** Error reading or writing to the filesystem. */
    public RepositoryFunctionException(IOException cause, Transience transience) {
      super(cause, transience);
    }

    public RepositoryFunctionException(EvalException cause, Transience transience) {
      super(cause, transience);
    }

    public RepositoryFunctionException(
        AlreadyReportedRepositoryAccessException cause, Transience transience) {
      super(cause, transience);
    }
  }

  /**
   * Encapsulates the exceptions that arise when accessing a repository. Error reporting should ONLY
   * be handled in {@link RepositoryDelegatorFunction#fetchRepository}.
   */
  public static class AlreadyReportedRepositoryAccessException extends AlreadyReportedException {
    public AlreadyReportedRepositoryAccessException(Exception e) {
      super(e.getMessage(), e.getCause());
      checkState(
          e instanceof NoSuchPackageException
              || e instanceof IOException
              || e instanceof EvalException
              || e instanceof ExternalPackageException,
          e);
    }
  }

  /**
   * Fetch the remote repository represented by the given rule.
   *
   * <p>When this method is called, it has already been determined that the repository is stale and
   * that it needs to be re-fetched.
   *
   * <p>The {@code env} argument can be used to fetch Skyframe dependencies the repository
   * implementation needs on the following conditions:
   *
   * <ul>
   *   <li>When a Skyframe value is missing, fetching must be restarted, thus, in order to avoid
   *       doing duplicate work, it's better to first request the Skyframe dependencies you need and
   *       only then start doing anything costly.
   *   <li>The output directory must be populated from within this method (and not from within
   *       another SkyFunction). This is because if it was populated in another SkyFunction, the
   *       repository function would be restarted <b>after</b> that SkyFunction has been run, and it
   *       would wipe the output directory clean.
   * </ul>
   */
  @ThreadSafe
  @Nullable
  public abstract FetchResult fetch(
      Rule rule, Path outputDirectory, BlazeDirectories directories, Environment env, SkyKey key)
      throws InterruptedException, RepositoryFunctionException;

  /** Whether the fetched repo contents are reproducible, hence cacheable. */
  public enum Reproducibility {
    YES,
    NO
  }

  /**
   * The result of the {@link #fetch} method.
   *
   * @param recordedInputValues Any recorded inputs (and their values) encountered during the fetch
   *     of the repo. Changes to these inputs will result in the repo being refetched in the future.
   *     Not an ImmutableMap, because regrettably the values can be null sometimes.
   * @param reproducible Whether the fetched repo contents are reproducible, hence cacheable.
   */
  public record FetchResult(
      Map<? extends RepoRecordedInput, String> recordedInputValues, Reproducibility reproducible) {}

  /**
   * We sometimes get to the point before fetching a repo when we have <em>just</em> fetched it,
   * particularly because of Skyframe restarts very late into RepositoryDelegatorFunction. In that
   * case, this method can be used to report that fact.
   */
  public abstract boolean wasJustFetched(Environment env);

  public static RootedPath getRootedPathFromLabel(Label label, Environment env)
      throws InterruptedException, EvalException {
    SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
    if (pkgLookupValue == null) {
      throw new NeedsSkyframeRestartException();
    }
    if (!pkgLookupValue.packageExists()) {
      String message = pkgLookupValue.getErrorMsg();
      if (pkgLookupValue == PackageLookupValue.NO_BUILD_FILE_VALUE) {
        message = PackageLookupFunction.explainNoBuildFileValue(label.getPackageIdentifier(), env);
      }
      throw Starlark.errorf("Unable to load package for %s: %s", label, message);
    }

    // And now for the file
    Root packageRoot = pkgLookupValue.getRoot();
    return RootedPath.toRootedPath(packageRoot, label.toPathFragment());
  }

  /**
   * Returns the values of the environment variables in the given set as an immutable map while
   * registering them as dependencies.
   */
  @Nullable
  public static ImmutableMap<String, Optional<String>> getEnvVarValues(
      Environment env, Set<String> keys) throws InterruptedException {
    Map<String, Optional<String>> environ = ActionEnvironmentFunction.getEnvironmentView(env, keys);
    if (environ == null) {
      return null;
    }
    Map<String, String> repoEnvOverride = PrecomputedValue.REPO_ENV.get(env);
    if (repoEnvOverride == null) {
      return null;
    }

    // Only depend on --repo_env values that are specified in keys.
    ImmutableMap.Builder<String, Optional<String>> repoEnv = ImmutableMap.builder();
    repoEnv.putAll(environ);
    for (String key : keys) {
      String value = repoEnvOverride.get(key);
      if (value != null) {
        repoEnv.put(key, Optional.of(value));
      }
    }
    return repoEnv.buildKeepingLast();
  }

  /**
   * Whether fetching is done using local operations only.
   *
   * <p>If this is false, Bazel may decide not to re-fetch the repository, for example when the
   * {@code --nofetch} command line option is used.
   */
  protected abstract boolean isLocal(Rule rule);

  /** Whether the rule declares it inspects the local environment for configure purpose. */
  protected abstract boolean isConfigure(Rule rule);

  protected static Path getExternalRepositoryDirectory(BlazeDirectories directories) {
    return directories.getOutputBase().getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
  }
}
