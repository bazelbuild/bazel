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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.rules.python.PythonVersion;
import com.google.devtools.build.lib.rules.python.PythonVersionTransition;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

/** Static utilities for managing output directory symlinks. */
public final class OutputDirectoryLinksUtils {

  // Static utilities class.
  private OutputDirectoryLinksUtils() {}

  /** Represents a single kind of convenience symlink ({@code bazel-bin}, etc.). */
  interface SymlinkDefinition {
    /**
     * Returns the name for this symlink in the workspace.
     *
     * <p>Note that this is independent of the target configuration(s) that may help determine the
     * symlink's destination.
     */
    String getLinkName(String symlinkPrefix, String productName, String workspaceBaseName);

    /**
     * Returns a list of candidate destination paths for the symlink.
     *
     * <p>The symlink should only be created if there is exactly one candidate.
     *
     * <p>{@code configGetter} is used to compute derived configurations, if needed. It is used for
     * symlinks that link to the output directories of configs that are related to, but not included
     * in, {@code targetConfigs}.
     */
    Set<Path> getLinkPaths(
        Set<BuildConfiguration> targetConfigs,
        Function<BuildOptions, BuildConfiguration> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot);
  }

  private static final class ConfigSymlink implements SymlinkDefinition {
    @FunctionalInterface
    private static interface ConfigPathGetter {
      ArtifactRoot apply(BuildConfiguration configuration, RepositoryName repositoryName);
    }

    private final String suffix;
    private final ConfigPathGetter configToRoot;

    public ConfigSymlink(String suffix, ConfigPathGetter configToRoot) {
      this.suffix = suffix;
      this.configToRoot = configToRoot;
    }

    @Override
    public String getLinkName(String symlinkPrefix, String productName, String workspaceBaseName) {
      return symlinkPrefix + suffix;
    }

    @Override
    public Set<Path> getLinkPaths(
        Set<BuildConfiguration> targetConfigs,
        Function<BuildOptions, BuildConfiguration> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot) {
      return targetConfigs.stream()
          .map(config -> configToRoot.apply(config, repositoryName).getRoot().asPath())
          .distinct()
          .collect(toImmutableSet());
    }
  }

  private enum ExecRootSymlink implements SymlinkDefinition {
    INSTANCE;

    @Override
    public String getLinkName(String symlinkPrefix, String productName, String workspaceBaseName) {
      return symlinkPrefix + workspaceBaseName;
    }

    @Override
    public Set<Path> getLinkPaths(
        Set<BuildConfiguration> targetConfigs,
        Function<BuildOptions, BuildConfiguration> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot) {
      return ImmutableSet.of(execRoot);
    }
  }

  private enum OutputSymlink implements SymlinkDefinition {
    PRODUCT_NAME {
      @Override
      public String getLinkName(
          String symlinkPrefix, String productName, String workspaceBaseName) {
        // TODO(b/35234395): This symlink is created for backwards compatibility, remove it once
        // we're sure it won't cause any other issues.
        return productName + "-out";
      }
    },
    SYMLINK_PREFIX {
      @Override
      public String getLinkName(
          String symlinkPrefix, String productName, String workspaceBaseName) {
        return symlinkPrefix + "out";
      }
    };

    @Override
    public Set<Path> getLinkPaths(
        Set<BuildConfiguration> targetConfigs,
        Function<BuildOptions, BuildConfiguration> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot) {
      return ImmutableSet.of(outputPath);
    }
  }

  private enum Py2BinSymlink implements SymlinkDefinition {
    INSTANCE;

    private static final PythonVersionTransition py2Transition =
        PythonVersionTransition.toConstant(PythonVersion.PY2);

    @Override
    public String getLinkName(String symlinkPrefix, String productName, String workspaceBaseName) {
      return symlinkPrefix + "py2-bin";
    }

    @Override
    public Set<Path> getLinkPaths(
        Set<BuildConfiguration> targetConfigs,
        Function<BuildOptions, BuildConfiguration> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot) {
      return targetConfigs.stream()
          .map(config -> configGetter.apply(py2Transition.patch(config.getOptions())))
          .map(config -> config.getBinDirectory(repositoryName).getRoot().asPath())
          .distinct()
          .collect(toImmutableSet());
    }
  }

  /**
   * Returns the (types of) convenience symlinks that should be created for the given options.
   *
   * <p>The result is always a subset of {@link #getAllLinkDefinitions}.
   */
  private static ImmutableList<SymlinkDefinition> getLinkDefinitions(
      boolean includeGenfiles, boolean includePy2Bin) {
    // The order of this list controls priority for PathPrettyPrinter#getPrettyPath.
    ImmutableList.Builder<SymlinkDefinition> builder = ImmutableList.builder();
    builder.add(new ConfigSymlink("bin", BuildConfiguration::getBinDirectory));
    if (includePy2Bin) {
      builder.add(Py2BinSymlink.INSTANCE);
    }
    builder.add(new ConfigSymlink("testlogs", BuildConfiguration::getTestLogsDirectory));
    if (includeGenfiles) {
      builder.add(new ConfigSymlink("genfiles", BuildConfiguration::getGenfilesDirectory));
    }
    builder.add(OutputSymlink.PRODUCT_NAME);
    builder.add(OutputSymlink.SYMLINK_PREFIX);
    builder.add(ExecRootSymlink.INSTANCE);
    return builder.build();
  }

  /**
   * Returns all (types of) convenience symlinks that may be created, independent of which ones are
   * actually requested by the build options.
   */
  private static final ImmutableList<SymlinkDefinition> getAllLinkDefinitions() {
    return getLinkDefinitions(/*includeGenfiles=*/ true, /*includePy2Bin=*/ true);
  }

  private static final String NO_CREATE_SYMLINKS_PREFIX = "/";

  public static Iterable<String> getOutputSymlinkNames(String productName, String symlinkPrefix) {
    ImmutableSet.Builder<String> builder = ImmutableSet.<String>builder();
    for (OutputSymlink definition : OutputSymlink.values()) {
      builder.add(definition.getLinkName(symlinkPrefix, productName, null));
    }
    return builder.build();
  }

  /**
   * Attempts to create convenience symlinks in the workspaceDirectory and in execRoot to the output
   * area and to the configuration-specific output directories. Issues a warning if it fails, e.g.
   * because workspaceDirectory is readonly.
   *
   * <p>Configuration-specific output symlinks will be created or updated if and only if the set of
   * {@code targetConfigs} contains only configurations whose output directories match. Otherwise -
   * i.e., if there are multiple configurations with distinct output directories or there were no
   * targets with non-null configurations in the build - any stale symlinks left over from previous
   * invocations will be removed.
   */
  static void createOutputDirectoryLinks(
      String workspaceName,
      Path workspace,
      Path execRoot,
      Path outputPath,
      EventHandler eventHandler,
      Set<BuildConfiguration> targetConfigs,
      Function<BuildOptions, BuildConfiguration> configGetter,
      String symlinkPrefix,
      String productName,
      boolean createGenfilesSymlink,
      boolean createPy2BinSymlink) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }

    List<String> failures = new ArrayList<>();
    List<String> ambiguousLinks = new ArrayList<>();
    Set<String> createdLinks = new LinkedHashSet<>();
    String workspaceBaseName = workspace.getBaseName();
    RepositoryName repositoryName = RepositoryName.createFromValidStrippedName(workspaceName);

    List<SymlinkDefinition> defs =
        getLinkDefinitions(
            /*includeGenfiles=*/ createGenfilesSymlink, /*includePy2Bin=*/ createPy2BinSymlink);
    for (SymlinkDefinition definition : defs) {
      String symlinkName = definition.getLinkName(symlinkPrefix, productName, workspaceBaseName);
      if (!createdLinks.add(symlinkName)) {
        // already created a link by this name
        continue;
      }
      Set<Path> candidatePaths =
          definition.getLinkPaths(
              targetConfigs, configGetter, repositoryName, outputPath, execRoot);
      if (candidatePaths.size() == 1) {
        createLink(workspace, symlinkName, Iterables.getOnlyElement(candidatePaths), failures);
      } else {
        removeLink(workspace, symlinkName, failures);
        // candidatePaths can be empty if the symlink decided not to be created. This can happen if,
        // say, py2-bin is enabled but there's an error producing the py2 configuration. In that
        // case, don't trigger a warning about an ambiguous link.
        if (candidatePaths.size() > 1) {
          ambiguousLinks.add(symlinkName);
        }
      }
    }

    if (!failures.isEmpty()) {
      eventHandler.handle(Event.warn(String.format(
          "failed to create one or more convenience symlinks for prefix '%s':\n  %s",
          symlinkPrefix, Joiner.on("\n  ").join(failures))));
    }
    if (!ambiguousLinks.isEmpty()) {
      eventHandler.handle(
          Event.warn(
              String.format(
                  "cleared convenience symlink(s) %s because their destinations would be ambiguous",
                  Joiner.on(", ").join(ambiguousLinks))));
    }
  }

  public static PathPrettyPrinter getPathPrettyPrinter(
      String symlinkPrefix, String productName, Path workspaceDirectory, Path workingDirectory) {
    return new PathPrettyPrinter(
        getAllLinkDefinitions(), symlinkPrefix, productName, workspaceDirectory, workingDirectory);
  }

  /**
   * Attempts to remove the convenience symlinks in the workspace directory.
   *
   * <p>Issues a warning if it fails, e.g. because workspaceDirectory is readonly.
   * Also cleans up any child directories created by a custom prefix.
   *
   * @param workspace the runtime's workspace
   * @param eventHandler the error eventHandler
   * @param symlinkPrefix the symlink prefix which should be removed
   * @param productName the product name
   */
  public static void removeOutputDirectoryLinks(String workspaceName, Path workspace,
      EventHandler eventHandler, String symlinkPrefix, String productName) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }
    List<String> failures = new ArrayList<>();

    String workspaceBaseName = workspace.getBaseName();
    for (SymlinkDefinition link : getAllLinkDefinitions()) {
      removeLink(
          workspace, link.getLinkName(symlinkPrefix, productName, workspaceBaseName), failures);
    }

    FileSystemUtils.removeDirectoryAndParents(workspace, PathFragment.create(symlinkPrefix));
    if (!failures.isEmpty()) {
      eventHandler.handle(Event.warn(String.format(
          "failed to remove one or more convenience symlinks for prefix '%s':\n  %s", symlinkPrefix,
          Joiner.on("\n  ").join(failures))));
    }
  }

  /**
   * Helper to createOutputDirectoryLinks that creates a symlink from base + name to target.
   */
  private static boolean createLink(Path base, String name, Path target, List<String> failures) {
    try {
      FileSystemUtils.createDirectoryAndParents(target);
    } catch (IOException e) {
      failures.add(String.format("cannot create directory %s: %s",
          target.getPathString(), e.getMessage()));
      return false;
    }
    try {
      FileSystemUtils.ensureSymbolicLink(base.getRelative(name), target);
    } catch (IOException e) {
      failures.add(String.format("cannot create symbolic link %s -> %s:  %s",
          name, target.getPathString(), e.getMessage()));
      return false;
    }

    return true;
  }

  /**
   * Helper to removeOutputDirectoryLinks that removes one of the Blaze convenience symbolic links.
   */
  private static boolean removeLink(Path base, String name, List<String> failures) {
    Path link = base.getRelative(name);
    try {
      if (link.isSymbolicLink()) {
        ExecutionTool.logger.finest("Removing " + link);
        link.delete();
      }
      return true;
    } catch (IOException e) {
      failures.add(String.format("%s: %s", name, e.getMessage()));
      return false;
    }
  }
}
