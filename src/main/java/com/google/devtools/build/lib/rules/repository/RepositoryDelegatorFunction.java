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

import static com.google.devtools.build.lib.rules.repository.RepositoryDirectoryDirtinessChecker.managedDirectoriesExist;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleFormatter;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.repository.ExternalRuleNotFoundException;
import com.google.devtools.build.lib.repository.RepositoryFailedEvent;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

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
  // Managed directories mappings, pre-calculated and injected by SequencedSkyframeExecutor
  // before each command.
  private final ManagedDirectoriesKnowledge managedDirectoriesKnowledge;

  private final ExternalPackageHelper externalPackageHelper;

  private final Supplier<Map<String, String>> clientEnvironmentSupplier;

  public RepositoryDelegatorFunction(
      ImmutableMap<String, RepositoryFunction> handlers,
      @Nullable RepositoryFunction starlarkHandler,
      AtomicBoolean isFetch,
      Supplier<Map<String, String>> clientEnvironmentSupplier,
      BlazeDirectories directories,
      ManagedDirectoriesKnowledge managedDirectoriesKnowledge,
      ExternalPackageHelper externalPackageHelper) {
    this.handlers = handlers;
    this.starlarkHandler = starlarkHandler;
    this.isFetch = isFetch;
    this.clientEnvironmentSupplier = clientEnvironmentSupplier;
    this.directories = directories;
    this.managedDirectoriesKnowledge = managedDirectoriesKnowledge;
    this.externalPackageHelper = externalPackageHelper;
  }

  public static RepositoryDirectoryValue.Builder symlink(
      Path source, PathFragment destination, String userDefinedPath, Environment env)
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
    FileValue repositoryValue = RepositoryFunction.getRepositoryDirectory(source, env);
    if (repositoryValue == null) {
      // TODO(bazel-team): If this returns null, we unnecessarily recreate the symlink above on the
      // second execution.
      return null;
    }

    if (!repositoryValue.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "The repository's path is \"%s\" (absolute: \"%s\") "
                      + "but this directory does not exist.",
                  userDefinedPath, destination)),
          Transience.PERSISTENT);
    }

    // Check that the repository contains a WORKSPACE file.
    // It's important to check the real path, otherwise this looks under the "external/[repo]" path
    // and cause a Skyframe cycle in the lookup.
    FileValue workspaceFileValue =
        LocalRepositoryFunction.getWorkspaceFile(repositoryValue.realRootedPath(), env);
    if (workspaceFileValue == null) {
      return null;
    }

    if (!workspaceFileValue.exists()) {
      throw new RepositoryFunctionException(
          new IOException("No WORKSPACE file found in " + source), Transience.PERSISTENT);
    }

    return RepositoryDirectoryValue.builder().setPath(source);
  }

  private void setupRepositoryRoot(Path repoRoot) throws RepositoryFunctionException {
    try {
      repoRoot.deleteTree();
      Preconditions.checkNotNull(repoRoot.getParentDirectory()).createDirectoryAndParents();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();

    Map<RepositoryName, PathFragment> overrides = REPOSITORY_OVERRIDES.get(env);
    boolean doNotFetchUnconditionally =
        DONT_FETCH_UNCONDITIONALLY.equals(DEPENDENCY_FOR_UNCONDITIONAL_FETCHING.get(env));
    boolean needsConfiguring = false;

    Path repoRoot = RepositoryFunction.getExternalRepositoryDirectory(directories)
        .getRelative(repositoryName.strippedName());

    if (Preconditions.checkNotNull(overrides).containsKey(repositoryName)) {
      DigestWriter.clearMarkerFile(directories, repositoryName);
      return setupOverride(
          overrides.get(repositoryName), env, repoRoot, repositoryName.strippedName());
    }

    Rule rule;
    try {
      rule = getRepository(repositoryName, env);
    } catch (ExternalRuleNotFoundException e) {
      return RepositoryDirectoryValue.NO_SUCH_REPOSITORY_VALUE;
    }
    if (rule == null) {
      return null;
    }

    RepositoryFunction handler = getHandler(rule);
    if (handler == null) {
      // If we refer to a non repository rule then the repository does not exist.
      return RepositoryDirectoryValue.NO_SUCH_REPOSITORY_VALUE;
    }

    if (handler.isConfigure(rule)) {
      needsConfiguring =
          !DONT_FETCH_UNCONDITIONALLY.equals(DEPENDENCY_FOR_UNCONDITIONAL_CONFIGURING.get(env));
    }

    if (env.valuesMissing()) {
      return null;
    }
    ImmutableSet<PathFragment> managedDirectories =
        managedDirectoriesKnowledge.getManagedDirectories(repositoryName);
    DigestWriter digestWriter =
        new DigestWriter(
            directories,
            repositoryName,
            rule,
            managedDirectories);

    // Local repositories are fetched regardless of the marker file because the operation is
    // generally fast and they do not depend on non-local data, so it does not make much sense to
    // try to cache them from across server instances.
    boolean fetchLocalRepositoryAlways = isFetch.get() && handler.isLocal(rule);
    if (!fetchLocalRepositoryAlways
        && managedDirectoriesExist(directories.getWorkspace(), managedDirectories)) {
      // For the non-local repositories, check if they are already up-to-date:
      // 1) unconditional fetching is not enabled, AND
      // 2) unconditional syncing is not enabled or the rule is not a configure rule, AND
      // 3) repository directory exists, AND
      // 4) marker file correctly describes the current repository state, AND
      // 5) managed directories, mapped to the repository, exist
      if (!needsConfiguring && doNotFetchUnconditionally && repoRoot.exists()) {
        byte[] markerHash = digestWriter.areRepositoryAndMarkerFileConsistent(handler, env);
        if (env.valuesMissing()) {
          return null;
        }
        if (markerHash != null) {
          // Now that we know that it exists and that we should not fetch unconditionally, we can
          // declare a Skyframe dependency on the repository root.
          RepositoryFunction.getRepositoryDirectory(repoRoot, env);
          if (env.valuesMissing()) {
            return null;
          }
          return RepositoryDirectoryValue.builder()
              .setPath(repoRoot)
              .setDigest(markerHash)
              .setManagedDirectories(managedDirectories)
              .build();
        }
      }
    }

    if (isFetch.get()) {
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
      return builder.setDigest(digest).setManagedDirectories(managedDirectories).build();
    }

    if (!repoRoot.exists()) {
      // The repository isn't on the file system, there is nothing we can do.
      throw new RepositoryFunctionException(
          new IOException("to fix, run\n\tbazel fetch //...\nExternal repository " + repositoryName
              + " not found and fetching repositories is disabled."),
          Transience.TRANSIENT);
    }

    // Declare a Skyframe dependency so that this is re-evaluated when something happens to the
    // directory.
    RepositoryFunction.getRepositoryDirectory(repoRoot, env);
    if (env.valuesMissing()) {
      return null;
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
        .setManagedDirectories(managedDirectories)
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

  private RepositoryDirectoryValue.Builder fetchRepository(
      SkyKey skyKey,
      Path repoRoot,
      Environment env,
      Map<String, String> markerData,
      RepositoryFunction handler,
      Rule rule)
      throws SkyFunctionException, InterruptedException {

    setupRepositoryRoot(repoRoot);

    String repositoryName = ((RepositoryName) skyKey.argument()).getName();
    env.getListener().post(new RepositoryFetching(repositoryName, false));

    RepositoryDirectoryValue.Builder repoBuilder;
    try {
      repoBuilder = handler.fetch(rule, repoRoot, directories, env, markerData, skyKey);
    } catch (SkyFunctionException e) {
      // Upon an exceptional exit, the fetching of that repository is over as well.
      env.getListener().post(new RepositoryFetching(repositoryName, true));
      env.getListener().post(new RepositoryFailedEvent(repositoryName, e.getMessage()));
      throw e;
    }

    if (env.valuesMissing()) {
      env.getListener().post(new RepositoryFetching(repositoryName, false, "Restarting."));
      return null;
    }
    env.getListener().post(new RepositoryFetching(repositoryName, true));
    return Preconditions.checkNotNull(repoBuilder);
  }

  /**
   * Uses a remote repository name to fetch the corresponding Rule describing how to get it. This
   * should be called from {@link SkyFunction#compute} functions, which should return null if this
   * returns null.
   */
  @Nullable
  private Rule getRepository(RepositoryName repositoryName, Environment env)
      throws ExternalPackageException, InterruptedException {
    return externalPackageHelper.getRuleByName(repositoryName.strippedName(), env);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private RepositoryDirectoryValue setupOverride(
      PathFragment sourcePath, Environment env, Path repoRoot, String pathAttr)
      throws RepositoryFunctionException, InterruptedException {
    setupRepositoryRoot(repoRoot);
    RepositoryDirectoryValue.Builder directoryValue = symlink(repoRoot, sourcePath, pathAttr, env);
    if (directoryValue == null) {
      return null;
    }
    byte[] digest = new byte[] {};
    return directoryValue.setDigest(digest).build();
  }

  // Escape a value for the marker file
  @VisibleForTesting
  static String escape(String str) {
    return str == null ? "\\0" : str.replace("\\", "\\\\").replace("\n", "\\n").replace(" ", "\\s");
  }

  // Unescape a value from the marker file
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
    private static final String MANAGED_DIRECTORIES_MARKER = "$MANAGED";
    private final Path markerPath;
    private final Rule rule;
    private final Map<String, String> markerData;
    private final String ruleKey;

    DigestWriter(
        BlazeDirectories directories,
        RepositoryName repositoryName,
        Rule rule,
        ImmutableSet<PathFragment> managedDirectories) {
      ruleKey = computeRuleKey(rule);
      markerPath = getMarkerPath(directories, repositoryName.strippedName());
      this.rule = rule;
      markerData = Maps.newHashMap();

      List<PathFragment> directoriesList = Ordering.natural().sortedCopy(managedDirectories);
      String directoriesString =
          directoriesList.stream()
              .map(PathFragment::getPathString)
              .collect(Collectors.joining(" "));
      markerData.put(MANAGED_DIRECTORIES_MARKER, directoriesString);
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
        if (Preconditions.checkNotNull(ruleKey).equals(markerRuleKey)
            && Objects.equals(
                markerData.get(MANAGED_DIRECTORIES_MARKER),
                this.markerData.get(MANAGED_DIRECTORIES_MARKER))) {
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
        getMarkerPath(directories, repoName.strippedName()).delete();
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }
  }

  private static class RepositoryFetching implements FetchProgress {
    final String id;
    final boolean finished;
    final String message;

    RepositoryFetching(String name, boolean finished) {
      this.id = name;
      this.finished = finished;
      this.message = finished ? "finished." : "fetching";
    }

    RepositoryFetching(String name, boolean finished, String message) {
      this.id = name;
      this.finished = finished;
      this.message = message;
    }

    @Override
    public String getResourceIdentifier() {
      return id;
    }

    @Override
    public String getProgress() {
      return message;
    }

    @Override
    public boolean isFinished() {
      return finished;
    }
  }
}
