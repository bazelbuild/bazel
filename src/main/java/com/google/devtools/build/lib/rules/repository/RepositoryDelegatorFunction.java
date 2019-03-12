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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleFormatter;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageUtil;
import com.google.devtools.build.lib.repository.ExternalRuleNotFoundException;
import com.google.devtools.build.lib.repository.RepositoryFailedEvent;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
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
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
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
      new Precomputed<>(PrecomputedValue.Key.create("repository_overrides"));

  public static final Precomputed<String> DEPENDENCY_FOR_UNCONDITIONAL_FETCHING =
      new Precomputed<>(
          PrecomputedValue.Key.create("dependency_for_unconditional_repository_fetching"));

  public static final Precomputed<Optional<RootedPath>> RESOLVED_FILE_FOR_VERIFICATION =
      new Precomputed<>(
          PrecomputedValue.Key.create("resolved_file_for_external_repository_verification"));

  public static final Precomputed<Set<String>> OUTPUT_VERIFICATION_REPOSITORY_RULES =
      new Precomputed<>(PrecomputedValue.Key.create("output_verification_repository_rules"));

  public static final Precomputed<Optional<RootedPath>> RESOLVED_FILE_INSTEAD_OF_WORKSPACE =
      new Precomputed<>(PrecomputedValue.Key.create("resolved_file_instead_of_workspace"));

  public static final String DONT_FETCH_UNCONDITIONALLY = "";

  // The marker file version is inject in the rule key digest so the rule key is always different
  // when we decide to update the format.
  private static final int MARKER_FILE_VERSION = 3;

  // A special repository delegate used to handle Skylark remote repositories if present.
  public static final String SKYLARK_DELEGATE_NAME = "$skylark";

  // Mapping of rule class name to RepositoryFunction.
  private final ImmutableMap<String, RepositoryFunction> handlers;

  // Delegate function to handle skylark remote repositories
  private final RepositoryFunction skylarkHandler;

  // This is a reference to isFetch in BazelRepositoryModule, which tracks whether the current
  // command is a fetch. Remote repository lookups are only allowed during fetches.
  private final AtomicBoolean isFetch;

  private final BlazeDirectories directories;

  private final Supplier<Map<String, String>> clientEnvironmentSupplier;

  public RepositoryDelegatorFunction(
      ImmutableMap<String, RepositoryFunction> handlers,
      @Nullable RepositoryFunction skylarkHandler,
      AtomicBoolean isFetch,
      Supplier<Map<String, String>> clientEnvironmentSupplier,
      BlazeDirectories directories) {
    this.handlers = handlers;
    this.skylarkHandler = skylarkHandler;
    this.isFetch = isFetch;
    this.clientEnvironmentSupplier = clientEnvironmentSupplier;
    this.directories = directories;
  }

  private void setupRepositoryRoot(Path repoRoot) throws RepositoryFunctionException {
    try {
      FileSystemUtils.deleteTree(repoRoot);
      FileSystemUtils.createDirectoryAndParents(repoRoot.getParentDirectory());
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Map<RepositoryName, PathFragment> overrides = REPOSITORY_OVERRIDES.get(env);
    if (env.valuesMissing()) {
      return null;
    }
    String fetchUnconditionally = DEPENDENCY_FOR_UNCONDITIONAL_FETCHING.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    Path repoRoot = RepositoryFunction.getExternalRepositoryDirectory(directories)
        .getRelative(repositoryName.strippedName());
    Path markerPath = getMarkerPath(directories, repositoryName.strippedName());
    if (overrides.containsKey(repositoryName)) {
      return setupOverride(
          repositoryName, overrides.get(repositoryName), env, repoRoot, markerPath);
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
    boolean hasNeedsUpdate = rule.getRuleClassObject().getNeedsUpdateFunction() != null;
    if (env.valuesMissing()) {
      return null;
    }

    RepositoryFunction handler;
    if (rule.getRuleClassObject().isSkylark()) {
      handler = skylarkHandler;
    } else {
      handler = handlers.get(rule.getRuleClass());
    }
    if (handler == null) {
      // If we refer to a non repository rule then the repository does not exist.
      return RepositoryDirectoryValue.NO_SUCH_REPOSITORY_VALUE;
    }

    handler.setClientEnvironment(clientEnvironmentSupplier.get());

    byte[] ruleSpecificData = handler.getRuleSpecificMarkerData(rule, env);
    if (ruleSpecificData == null) {
      return null;
    }
    String ruleKey = computeRuleKey(rule, ruleSpecificData);
    Map<String, String> markerData = new TreeMap<>();
    if (env.valuesMissing()) {
      return null;
    }
    boolean isLocal = handler.isLocal(env, repoRoot.getFileSystem(), rule);
    if (env.valuesMissing()) {
      return null;
    }
    if (isFetch.get() && isLocal) {
      // Local repositories are fetched regardless of the marker file because the operation is
      // generally fast and they do not depend on non-local data, so it does not make much sense to
      // try to cache from across server instances.
      setupRepositoryRoot(repoRoot);
      env.getListener().post(new RepositoryFetching(repositoryName.getName(), false));
      RepositoryDirectoryValue.Builder localRepo =
          handler.fetch(rule, repoRoot, directories, env, markerData, skyKey);
      if (localRepo == null) {
        return null;
      } else {
        // We write the marker file for local repository essentially for getting the digest and
        // injecting it in the RepositoryDirectoryValue.
        byte[] digest = writeMarkerFile(markerPath, markerData, ruleKey);
        env.getListener().post(new RepositoryFetching(repositoryName.getName(), true));
        return localRepo.setHasRefreshRoots(hasNeedsUpdate).setDigest(digest).build();
      }
    }

    // We check the repository root for existence here, but we can't depend on the FileValue,
    // because it's possible that we eventually create that directory in which case the FileValue
    // and the state of the file system would be inconsistent.

    byte[] markerHash = isFilesystemUpToDate(markerPath, rule, ruleKey, handler, env);
    if (env.valuesMissing()) {
      return null;
    }
    if (DONT_FETCH_UNCONDITIONALLY.equals(fetchUnconditionally)
        && markerHash != null
        && repoRoot.exists()) {
      // Now that we know that it exists and that we should not fetch unconditionally, we can
      // declare a Skyframe dependency on the repository root.
      RepositoryFunction.getRepositoryDirectory(repoRoot, env);
      if (env.valuesMissing()) {
        return null;
      }

      return RepositoryDirectoryValue.builder().setPath(repoRoot).setDigest(markerHash).setHasRefreshRoots(hasNeedsUpdate).build();
    }

    if (isFetch.get()) {
      // Fetching enabled, go ahead.
      env.getListener().post(new RepositoryFetching(repositoryName.getName(), false));
      setupRepositoryRoot(repoRoot);
      RepositoryDirectoryValue.Builder result = null;
      try {
        result = handler.fetch(rule, repoRoot, directories, env, markerData, skyKey);
      } catch (SkyFunctionException e) {
        // Upon an exceptional exit, the fetching of that repository is over as well.
        env.getListener().post(new RepositoryFetching(repositoryName.getName(), true));
        env.getListener().post(new RepositoryFailedEvent(repositoryName.strippedName()));
        throw e;
      }
      if (env.valuesMissing()) {
        env.getListener()
            .post(new RepositoryFetching(repositoryName.getName(), false, "Restarting."));
        return null;
      }
      env.getListener().post(new RepositoryFetching(repositoryName.getName(), true));

      // No new Skyframe dependencies must be added between calling the repository implementation
      // and writing the marker file because if they aren't computed, it would cause a Skyframe
      // restart thus calling the possibly very slow (networking, decompression...) fetch()
      // operation again. So we write the marker file here immediately.
      byte[] digest = writeMarkerFile(markerPath, markerData, ruleKey);
      return result.setDigest(digest).setHasRefreshRoots(hasNeedsUpdate).build();
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
    FileValue repoRootValue = RepositoryFunction.getRepositoryDirectory(repoRoot, env);
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

    return RepositoryDirectoryValue.builder().setPath(repoRootValue.realRootedPath().asPath())
        .setFetchingDelayed().setHasRefreshRoots(hasNeedsUpdate).build();
  }

  /**
   * Uses a remote repository name to fetch the corresponding Rule describing how to get it. This
   * should be called from {@link SkyFunction#compute} functions, which should return null if this
   * returns null.
   */
  @Nullable
  private static Rule getRepository(
      RepositoryName repositoryName, Environment env)
      throws ExternalPackageException, InterruptedException {
    return ExternalPackageUtil.getRuleByName(repositoryName.strippedName(), env);
  }

  private String computeRuleKey(Rule rule, byte[] ruleSpecificData) {
    return new Fingerprint().addBytes(RuleFormatter.serializeRule(rule).build().toByteArray())
        .addBytes(ruleSpecificData)
        .addInt(MARKER_FILE_VERSION).hexDigestAndReset();
  }

  /**
   * Checks if the state of the repository in the file system is consistent with the rule in the
   * WORKSPACE file.
   *
   * <p>
   * Deletes the marker file if not so that no matter what happens after, the state of the file
   * system stays consistent.
   *
   * <p>
   * Returns null if the file system is not up to date and a hash of the marker file if the file
   * system is up to date.
   */
  @Nullable
  private byte[] isFilesystemUpToDate(Path markerPath, Rule rule, String ruleKey,
      RepositoryFunction handler, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    try {
      if (!markerPath.exists()) {
        return null;
      }

      String content = FileSystemUtils.readContent(markerPath, StandardCharsets.UTF_8);

      String[] lines = content.split("\n");
      Map<String, String> markerData = new TreeMap<>();
      String markerRuleKey = "";
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
      boolean result = false;
      if (markerRuleKey.equals(ruleKey)) {
        result = handler.verifyMarkerData(rule, markerData, env);
        if (env.valuesMissing()) {
          return null;
        }
      }

      if (result) {
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
    StringBuffer result = new StringBuffer();
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

  private byte[] writeMarkerFile(
      Path markerPath, Map<String, String> markerData, String ruleKey)
      throws RepositoryFunctionException {
    try {
      StringBuilder builder = new StringBuilder();
      builder.append(ruleKey).append("\n");
      for (Map.Entry<String, String> data : markerData.entrySet()) {
        String key = data.getKey();
        String value = data.getValue();
        builder.append(escape(key)).append(" ").append(escape(value)).append("\n");
      }
      String content = builder.toString();
      FileSystemUtils.writeContent(markerPath, StandardCharsets.UTF_8, content);
      return new Fingerprint().addString(content).digestAndReset();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  private static Path getMarkerPath(BlazeDirectories directories, String ruleName) {
    return RepositoryFunction.getExternalRepositoryDirectory(directories)
        .getChild("@" + ruleName + ".marker");
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private RepositoryDirectoryValue setupOverride(
      RepositoryName repositoryName, PathFragment sourcePath, Environment env, Path repoRoot,
      Path markerPath)
      throws RepositoryFunctionException, InterruptedException {
    setupRepositoryRoot(repoRoot);
    RepositoryDirectoryValue.Builder directoryValue = LocalRepositoryFunction.symlink(
        repoRoot, sourcePath, env);
    if (directoryValue == null) {
      return null;
    }
    String ruleKey = new Fingerprint().addBytes(repositoryName.strippedName().getBytes())
        .addBytes(repoRoot.getFileSystem().getPath(sourcePath).getPathString().getBytes())
        .addInt(MARKER_FILE_VERSION).hexDigestAndReset();
    byte[] digest = writeMarkerFile(markerPath, new TreeMap<String, String>(), ruleKey);
    return directoryValue.setDigest(digest).build();
  }

  private class RepositoryFetching implements FetchProgress {
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
