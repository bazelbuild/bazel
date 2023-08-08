// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.rules.repository;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleFormatter;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.repository.ExternalRuleNotFoundException;
import com.google.devtools.build.lib.repository.RepositoryFailedEvent;
import com.google.devtools.build.lib.repository.RepositoryFetchProgress;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue.NoRepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.AlreadyReportedRepositoryAccessException;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A {@link SkyFunction} that implements delegation to the correct repository fetcher.
 *
 * <p>
 * Each repository in the WORKSPACE file is represented by a {@link SkyValue} that is computed by
 * this function.
 */
public final class RepositoryDelegatorFunction implements SkyFunction {
  public static final Precomputed<Map<RepositoryName, PathFragment>> REPOSITORY_OVERRIDES =
      new Precomputed<>("repository_overrides");

  public static final Precomputed<String> DEPENDENCY_FOR_UNCONDITIONAL_FETCHING =
      new Precomputed<>("dependency_for_unconditional_repository_fetching");

  public static final Precomputed<String> DEPENDENCY_FOR_UNCONDITIONAL_CONFIGURING =
      new Precomputed<>("dependency_for_unconditional_configuring");

  public static final Precomputed<Optional<RootedPath>> RESOLVED_FILE_FOR_VERIFICATION =
      new Precomputed<>("resolved_file_for_external_repository_verification");

  public static final Precomputed<Set<String>> OUTPUT_VERIFICATION_REPOSITORY_RULES =
      new Precomputed<>("output_verification_repository_rules");

  public static final Precomputed<Optional<RootedPath>> RESOLVED_FILE_INSTEAD_OF_WORKSPACE =
      new Precomputed<>("resolved_file_instead_of_workspace");

  public static final String DONT_FETCH_UNCONDITIONALLY = "";

  // The marker file version is inject in the rule key digest so the rule key is always different
  // when we decide to update the format.
  private static final int MARKER_FILE_VERSION = 3;

  // Mapping of rule class name to RepositoryFunction.
  private final ImmutableMap<String, RepositoryFunction> handlers;

  // Delegate function to handle Starlark remote repositories
  private final RepositoryFunction starlarkHandler;

  // This is a reference to isFetch in BazelRepositoryModule, which tracks whether the current
  // command is a fetch. Remote repository lookups are only allowed during fetches.
  private final AtomicBoolean isFetch;

  private final BlazeDirectories directories;

  private final ExternalPackageHelper externalPackageHelper;

  private final Supplier<Map<String, String>> clientEnvironmentSupplier;

  public RepositoryDelegatorFunction(
      ImmutableMap<String, RepositoryFunction> handlers,
      @Nullable RepositoryFunction starlarkHandler,
      AtomicBoolean isFetch,
      Supplier<Map<String, String>> clientEnvironmentSupplier,
      BlazeDirectories directories,
      ExternalPackageHelper externalPackageHelper) {
    this.handlers = handlers;
    this.starlarkHandler = starlarkHandler;
    this.isFetch = isFetch;
    this.clientEnvironmentSupplier = clientEnvironmentSupplier;
    this.directories = directories;
    this.externalPackageHelper = externalPackageHelper;
  }

  @Nullable
  public static RepositoryDirectoryValue.Builder symlinkRepoRoot(
      BlazeDirectories directories,
      Path source,
      Path destination,
      String userDefinedPath,
      Environment env)
      throws RepositoryFunctionException, InterruptedException {
    try {
      source.createSymbolicLink(destination);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "Could not create symlink to repository \"%s\" (absolute path: \"%s\"): %s",
                  userDefinedPath, destination, e.getMessage()),
              e),
          Transience.TRANSIENT);
    }

    // Check that the target directory exists and is a directory.
    // Note that we have to check `destination` and not `source` here, otherwise we'd have a
    // circular dependency between SkyValues.
    RootedPath targetDirRootedPath;
    if (destination.startsWith(directories.getInstallBase())) {
      // The install base only changes with the Bazel binary so it's acceptable not to add its
      // ancestors as Skyframe dependencies.
      targetDirRootedPath =
          RootedPath.toRootedPath(Root.fromPath(destination), PathFragment.EMPTY_FRAGMENT);
    } else {
      targetDirRootedPath =
          RootedPath.toRootedPath(Root.absoluteRoot(destination.getFileSystem()), destination);
    }

    FileValue targetDirValue;
    try {
      targetDirValue =
          (FileValue) env.getValueOrThrow(FileValue.key(targetDirRootedPath), IOException.class);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not access " + destination + ": " + e.getMessage()),
          Transience.PERSISTENT);
    }
    if (targetDirValue == null) {
      // TODO(bazel-team): If this returns null, we unnecessarily recreate the symlink above on the
      // second execution.
      return null;
    }

    if (!targetDirValue.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "The repository's path is \"%s\" (absolute: \"%s\") "
                      + "but it does not exist or is not a directory.",
                  userDefinedPath, destination)),
          Transience.PERSISTENT);
    }

    // Check that the repository contains a WORKSPACE file.
    // Note that we need to do this here since we're not creating a WORKSPACE file ourselves, but
    // entrusting the entire contents of the repo root to this target directory.
    FileValue workspaceFileValue = getWorkspaceFile(targetDirRootedPath, env);
    if (workspaceFileValue == null) {
      return null;
    }

    if (!workspaceFileValue.exists()) {
      throw new RepositoryFunctionException(
          new IOException("No WORKSPACE file found in " + source), Transience.PERSISTENT);
    }
    return RepositoryDirectoryValue.builder().setPath(source);
  }

  @Nullable
  private static FileValue getWorkspaceFile(RootedPath directory, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    RootedPath workspaceRootedFile;
    try {
      workspaceRootedFile = WorkspaceFileHelper.getWorkspaceRootedFile(directory, env);
      if (workspaceRootedFile == null) {
        return null;
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Could not determine workspace file (\"WORKSPACE.bazel\" or \"WORKSPACE\"): "
                  + e.getMessage()),
          Transience.PERSISTENT);
    }
    SkyKey workspaceFileKey = FileValue.key(workspaceRootedFile);
    FileValue value;
    try {
      value = (FileValue) env.getValueOrThrow(workspaceFileKey, IOException.class);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not access " + workspaceRootedFile + ": " + e.getMessage()),
          Transience.PERSISTENT);
    }
    return value;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, RepositoryFunctionException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();

    if (!repositoryName.isVisible()) {
      return new NoRepositoryDirectoryValue(
          String.format(
              "No repository visible as '@%s' from %s",
              repositoryName.getName(), repositoryName.getOwnerRepoDisplayString()));
    }

    Map<RepositoryName, PathFragment> overrides = REPOSITORY_OVERRIDES.get(env);
    Path repoRoot =
        RepositoryFunction.getExternalRepositoryDirectory(directories)
            .getRelative(repositoryName.getName());

    if (Preconditions.checkNotNull(overrides).containsKey(repositoryName)) {
      DigestWriter.clearMarkerFile(directories, repositoryName);
      return setupOverride(overrides.get(repositoryName), env, repoRoot, repositoryName.getName());
    }

    Rule rule = null;
    if (starlarkSemantics.getBool(BuildLanguageOptions.ENABLE_BZLMOD)) {
      // Tries to get a repository rule instance from Bzlmod generated repos.
      SkyKey key = BzlmodRepoRuleValue.key(repositoryName);
      BzlmodRepoRuleValue value = (BzlmodRepoRuleValue) env.getValue(key);

      if (env.valuesMissing()) {
        return null;
      }

      if (value != BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE) {
        rule = value.getRule();
      }
    }

    if (rule == null) {
      // fallback to look up the repository in the WORKSPACE file.
      try {
        rule = getRepoRuleFromWorkspace(repositoryName, env);
        if (env.valuesMissing()) {
          return null;
        }
      } catch (NoSuchRepositoryException e) {
        return new NoRepositoryDirectoryValue(
            String.format("Repository '%s' is not defined", repositoryName.getCanonicalForm()));
      }
    }

    RepositoryFunction handler = getHandler(rule);
    if (handler == null) {
      // If we refer to a non repository rule then the repository does not exist.
      return new NoRepositoryDirectoryValue(
          String.format("'%s' is not a repository rule", repositoryName.getCanonicalForm()));
    }

    boolean needsConfiguring = false;
    if (handler.isConfigure(rule)) {
      needsConfiguring =
          !DONT_FETCH_UNCONDITIONALLY.equals(DEPENDENCY_FOR_UNCONDITIONAL_CONFIGURING.get(env));
    }
    if (env.valuesMissing()) {
      return null;
    }

    DigestWriter digestWriter = new DigestWriter(directories, repositoryName, rule);
    boolean doNotFetchUnconditionally =
        DONT_FETCH_UNCONDITIONALLY.equals(DEPENDENCY_FOR_UNCONDITIONAL_FETCHING.get(env));

    // Local repositories are fetched regardless of the marker file because the operation is
    // generally fast and they do not depend on non-local data, so it does not make much sense to
    // try to cache them from across server instances.
    boolean fetchLocalRepositoryAlways = isFetch.get() && handler.isLocal(rule);
    if (!fetchLocalRepositoryAlways) {
      // For the non-local repositories, check if they are already up-to-date:
      // 1) unconditional fetching is not enabled, AND
      // 2) unconditional syncing is not enabled or the rule is not a configure rule, AND
      // 3) repository directory exists, AND
      // 4) marker file correctly describes the current repository state
      if (!needsConfiguring && doNotFetchUnconditionally && repoRoot.exists()) {
        byte[] markerHash = digestWriter.areRepositoryAndMarkerFileConsistent(handler, env);
        if (env.valuesMissing()) {
          return null;
        }
        if (markerHash != null) {
          return RepositoryDirectoryValue.builder()
              .setPath(repoRoot)
              .setDigest(markerHash)
              .build();
        }
      }
    }

    if (isFetch.get()) {
      // Fetching a repository is a long-running operation that can easily be interrupted. If it is
      // and the marker file exists on disk, a new call of this method may treat this repository as
      // valid even though it is in an inconsistent state. Clear the marker file and only recreate
      // it after fetching is done to prevent this scenario.
      DigestWriter.clearMarkerFile(directories, repositoryName);
      // Fetching enabled, go ahead.
      RepositoryDirectoryValue.Builder builder =
          fetchRepository(skyKey, repoRoot, env, digestWriter.getMarkerData(), handler, rule);
      if (builder == null) {
        return null;
      }

      // No new Skyframe dependencies must be added between calling the repository implementation
      // and writing the marker file because if they aren't computed, it would cause a Skyframe
      // restart thus calling the possibly very slow (networking, decompression...) fetch()
      // operation again. So we write the marker file here immediately.
      byte[] digest = digestWriter.writeMarkerFile();
      return builder.setDigest(digest).build();
    }

    if (!repoRoot.exists()) {
      // The repository isn't on the file system, there is nothing we can do.
      throw new RepositoryFunctionException(
          new IOException(
              "to fix, run\n\tbazel fetch //...\nExternal repository "
                  + repositoryName.getNameWithAt()
                  + " not found and fetching repositories is disabled."),
          Transience.TRANSIENT);
    }

    // Try to build with whatever is on the file system and emit a warning.
    env.getListener()
        .handle(Event.warn(rule.getLocation(),
            String.format(
                "External repository '%s' is not up-to-date and fetching is disabled. To update, "
                    + "run the build without the '--nofetch' command line option.",
                rule.getName())));

    return RepositoryDirectoryValue.builder()
        .setPath(repoRoot)
        .setFetchingDelayed()
        .setDigest(new Fingerprint().digestAndReset())
        .build();
  }

  private RepositoryFunction getHandler(Rule rule) {
    RepositoryFunction handler;
    if (rule.getRuleClassObject().isStarlark()) {
      handler = starlarkHandler;
    } else {
      handler = handlers.get(rule.getRuleClass());
    }
    if (handler != null) {
      handler.setClientEnvironment(clientEnvironmentSupplier.get());
    }

    return handler;
  }

  @Nullable
  private RepositoryDirectoryValue.Builder fetchRepository(
      SkyKey skyKey,
      Path repoRoot,
      Environment env,
      Map<String, String> markerData,
      RepositoryFunction handler,
      Rule rule)
      throws InterruptedException, RepositoryFunctionException {

    handler.setupRepoRootBeforeFetching(repoRoot);

    RepositoryName repoName = (RepositoryName) skyKey.argument();
    env.getListener().post(RepositoryFetchProgress.ongoing(repoName, "starting"));

    RepositoryDirectoryValue.Builder repoBuilder;
    try {
      repoBuilder = handler.fetch(rule, repoRoot, directories, env, markerData, skyKey);
    } catch (RepositoryFunctionException e) {
      // Upon an exceptional exit, the fetching of that repository is over as well.
      env.getListener().post(RepositoryFetchProgress.finished(repoName));
      env.getListener().post(new RepositoryFailedEvent(repoName, e.getMessage()));
      env.getListener()
          .handle(
              Event.error(
                  rule.getLocation(), String.format("fetching %s: %s", rule, e.getMessage())));

      // Rewrap the underlying exception to signal callers not to re-report this error.
      throw new RepositoryFunctionException(
          new AlreadyReportedRepositoryAccessException(e.getCause()),
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }

    if (env.valuesMissing()) {
      handler.reportSkyframeRestart(env, repoName);
      return null;
    }
    env.getListener().post(RepositoryFetchProgress.finished(repoName));
    return Preconditions.checkNotNull(repoBuilder);
  }

  /**
   * Uses a remote repository name to fetch the corresponding Rule describing how to get it. This
   * should be called from {@link SkyFunction#compute} functions, which should return null if this
   * returns null.
   */
  @Nullable
  private Rule getRepoRuleFromWorkspace(RepositoryName repositoryName, Environment env)
      throws InterruptedException, RepositoryFunctionException, NoSuchRepositoryException {
    try {
      return externalPackageHelper.getRuleByName(repositoryName.getName(), env);
    } catch (ExternalRuleNotFoundException e) {
      // This is caught and handled immediately in compute().
      throw new NoSuchRepositoryException();
    } catch (ExternalPackageException e) {
      throw new RepositoryFunctionException(e);
    }
  }

  /** Marker exception for the case where a repository is not defined. */
  private static final class NoSuchRepositoryException extends Exception {}

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @Nullable
  private RepositoryDirectoryValue setupOverride(
      PathFragment sourcePath, Environment env, Path repoRoot, String pathAttr)
      throws RepositoryFunctionException, InterruptedException {
    RepositoryFunction.setupRepoRoot(repoRoot);
    RepositoryDirectoryValue.Builder directoryValue =
        symlinkRepoRoot(
            directories,
            repoRoot,
            directories.getWorkspace().getRelative(sourcePath),
            pathAttr,
            env);
    if (directoryValue == null) {
      return null;
    }
    byte[] digest = new Fingerprint().addPath(sourcePath).digestAndReset();
    return directoryValue.setDigest(digest).build();
  }

  // Escape a value for the marker file
  @VisibleForTesting
  static String escape(String str) {
    return str == null ? "\\0" : str.replace("\\", "\\\\").replace("\n", "\\n").replace(" ", "\\s");
  }

  // Unescape a value from the marker file
  @Nullable
  @VisibleForTesting
  static String unescape(String str) {
    if (str.equals("\\0")) {
      return null; // \0 == null string
    }
    StringBuilder result = new StringBuilder();
    boolean escaped = false;
    for (int i = 0; i < str.length(); i++) {
      char c = str.charAt(i);
      if (escaped) {
        if (c == 'n') {  // n means new line
          result.append("\n");
        } else if (c == 's') { // s means space
          result.append(" ");
        } else {  // Any other escaped characters are just un-escaped
          result.append(c);
        }
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else {
        result.append(c);
      }
    }
    return result.toString();
  }

  private static class DigestWriter {
    private final Path markerPath;
    private final Rule rule;
    // not just Map<> to signal that iteration order must be deterministic
    private final TreeMap<String, String> markerData;
    private final String ruleKey;

    DigestWriter(BlazeDirectories directories, RepositoryName repositoryName, Rule rule) {
      ruleKey = computeRuleKey(rule);
      markerPath = getMarkerPath(directories, repositoryName.getName());
      this.rule = rule;
      markerData = Maps.newTreeMap();
    }

    byte[] writeMarkerFile() throws RepositoryFunctionException {
      StringBuilder builder = new StringBuilder();
      builder.append(ruleKey).append("\n");
      for (Map.Entry<String, String> data : markerData.entrySet()) {
        String key = data.getKey();
        String value = data.getValue();
        builder.append(escape(key)).append(" ").append(escape(value)).append("\n");
      }
      String content = builder.toString();
      try {
        FileSystemUtils.writeContent(markerPath, StandardCharsets.UTF_8, content);
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
      return new Fingerprint().addString(content).digestAndReset();
    }

    /**
     * Checks if the state of the repository in the file system is consistent with the rule in the
     * WORKSPACE file.
     *
     * <p>Deletes the marker file if not so that no matter what happens after, the state of the file
     * system stays consistent.
     *
     * <p>Returns null if the file system is not up to date and a hash of the marker file if the
     * file system is up to date.
     *
     * <p>We check the repository root for existence here, but we can't depend on the FileValue,
     * because it's possible that we eventually create that directory in which case the FileValue
     * and the state of the file system would be inconsistent.
     */
    @Nullable
    byte[] areRepositoryAndMarkerFileConsistent(RepositoryFunction handler, Environment env)
        throws RepositoryFunctionException, InterruptedException {
      if (!markerPath.exists()) {
        return null;
      }

      Map<String, String> markerData = new TreeMap<>();
      String content;
      try {
        content = FileSystemUtils.readContent(markerPath, StandardCharsets.UTF_8);
        String markerRuleKey = readMarkerFile(content, markerData);
        boolean verified = false;
        if (Preconditions.checkNotNull(ruleKey).equals(markerRuleKey)) {
          verified = handler.verifyMarkerData(rule, markerData, env);
          if (env.valuesMissing()) {
            return null;
          }
        }

        if (verified) {
          return new Fingerprint().addString(content).digestAndReset();
        } else {
          // So that we are in a consistent state if something happens while fetching the repository
          markerPath.delete();
          return null;
        }
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }

    Map<String, String> getMarkerData() {
      return markerData;
    }

    @Nullable
    private String readMarkerFile(String content, Map<String, String> markerData) {
      String markerRuleKey = null;
      String[] lines = content.split("\n");

      boolean firstLine = true;
      for (String line : lines) {
        if (firstLine) {
          markerRuleKey = line;
          firstLine = false;
        } else {
          int sChar = line.indexOf(' ');
          String key = line;
          String value = "";
          if (sChar > 0) {
            key = unescape(line.substring(0, sChar));
            value = unescape(line.substring(sChar + 1));
          }
          markerData.put(key, value);
        }
      }
      return markerRuleKey;
    }

    private String computeRuleKey(Rule rule) {
      return new Fingerprint()
          .addBytes(RuleFormatter.serializeRule(rule).build().toByteArray())
          .addInt(MARKER_FILE_VERSION)
          .hexDigestAndReset();
    }

    private static Path getMarkerPath(BlazeDirectories directories, String ruleName) {
      return RepositoryFunction.getExternalRepositoryDirectory(directories)
          .getChild("@" + ruleName + ".marker");
    }

    static void clearMarkerFile(BlazeDirectories directories, RepositoryName repoName)
        throws RepositoryFunctionException {
      try {
        getMarkerPath(directories, repoName.getName()).delete();
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }
  }

}
