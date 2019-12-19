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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConvenienceSymlinks;
import com.google.devtools.build.lib.analysis.config.ConvenienceSymlinks.OutputSymlink;
import com.google.devtools.build.lib.analysis.config.ConvenienceSymlinks.SymlinkDefinition;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions.ConvenienceSymlinksMode;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
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

  /**
   * Returns all (types of) convenience symlinks that may be created.
   *
   * <p>Note that this is independent of which symlinks are actually requested by the build options;
   * that's controlled by returning no candidates in {@link SymlinkDefinition#getLinkPaths}.
   *
   * <p>The order of the result indicates precedence for {@link PathPrettyPrinter}.
   */
  private static final ImmutableList<SymlinkDefinition> getAllLinkDefinitions(
      Iterable<SymlinkDefinition> symlinkDefinitions) {
    ImmutableList.Builder<SymlinkDefinition> builder = ImmutableList.builder();
    builder.addAll(ConvenienceSymlinks.getStandardLinkDefinitions());
    builder.addAll(symlinkDefinitions);
    return builder.build();
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
   * Attempts to create or delete convenience symlinks in the workspace to the various output
   * directories, and generates associated log events.
   *
   * <p>If {@code --symlink_prefix} is {@link NO_CREATE_SYMLINKS_PREFIX}, or {@code
   * --experimental_convenience_symlinks} is {@link ConvenienceSymlinksMode.IGNORE}, this method is
   * a no-op.
   *
   * <p>Otherwise, for each symlink type, we decide whether the symlink should exist or not. If it
   * should exist, it is created with the appropriate destination path; if not, it is deleted if
   * already present on the file system. In either case, the decision of whether to create or delete
   * the symlink is logged. (Note that deleting pre-existing symlinks helps ensure the user's
   * workspace is in a consistent state after the build. However, if the {@code --symlink_prefix}
   * has changed, we have no way to cleanup old symlink names leftover from a previous invocation.)
   *
   * <p>If {@code --experimental_convenience_symlinks} is set to {@link
   * ConvenienceSymlinksMode.CLEAN}, all symlinks are set to be deleted. If it's set to {@link
   * ConvenienceSymlinksMode.NORMAL}, each symlink type decides whether it should be created or
   * deleted. (A symlink may decide to be deleted if e.g. it is disabled by a flag, or would want to
   * point to more than one destination.) If it's set to {@link ConvenienceSymlinksMode.LOG_ONLY},
   * the same logic is run as in the {@code NORMAL} case, but the result is only emitting log
   * messages, with no actual filesystem mutations.
   *
   * <p>A warning is emitted if a symlink would resolve to multiple destinations, or if a filesystem
   * mutation operation fails.
   */
  static void createOutputDirectoryLinks(
      Iterable<SymlinkDefinition> symlinkDefinitions,
      BuildRequestOptions buildRequestOptions,
      String workspaceName,
      Path workspace,
      Path execRoot,
      Path outputPath,
      EventHandler eventHandler,
      Set<BuildConfiguration> targetConfigs,
      Function<BuildOptions, BuildConfiguration> configGetter,
      String productName) {
    String symlinkPrefix = buildRequestOptions.getSymlinkPrefix(productName);
    ConvenienceSymlinksMode mode = buildRequestOptions.experimentalConvenienceSymlinks;
    if (mode == ConvenienceSymlinksMode.IGNORE || NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }

    List<String> failures = new ArrayList<>();
    List<String> ambiguousLinks = new ArrayList<>();
    Set<String> createdLinks = new LinkedHashSet<>();
    String workspaceBaseName = workspace.getBaseName();
    RepositoryName repositoryName = RepositoryName.createFromValidStrippedName(workspaceName);

    for (SymlinkDefinition symlink : getAllLinkDefinitions(symlinkDefinitions)) {
      String linkName = symlink.getLinkName(symlinkPrefix, productName, workspaceBaseName);
      if (!createdLinks.add(linkName)) {
        // already created a link by this name
        continue;
      }
      if (mode == ConvenienceSymlinksMode.CLEAN) {
        removeLink(workspace, linkName, failures);
      } else {
        Set<Path> candidatePaths =
            symlink.getLinkPaths(
                buildRequestOptions,
                targetConfigs,
                configGetter,
                repositoryName,
                outputPath,
                execRoot);
        if (candidatePaths.size() == 1) {
          createLink(workspace, linkName, Iterables.getOnlyElement(candidatePaths), failures);
        } else {
          removeLink(workspace, linkName, failures);
          // candidatePaths can be empty if the symlink decided not to be created. This can happen
          // if, say, py2-bin is enabled but there's an error producing the py2 configuration. In
          // that case, don't trigger a warning about an ambiguous link.
          if (candidatePaths.size() > 1) {
            ambiguousLinks.add(linkName);
          }
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
      Iterable<SymlinkDefinition> symlinkDefinitions,
      String symlinkPrefix,
      String productName,
      Path workspaceDirectory,
      Path workingDirectory) {
    return new PathPrettyPrinter(
        getAllLinkDefinitions(symlinkDefinitions),
        symlinkPrefix,
        productName,
        workspaceDirectory,
        workingDirectory);
  }

  /**
   * Attempts to remove the convenience symlinks in the workspace directory.
   *
   * <p>Issues a warning if it fails, e.g. because workspaceDirectory is readonly. Also cleans up
   * any child directories created by a custom prefix.
   *
   * @param symlinkDefinitions extra symlink types added by the {@link ConfiguredRuleClassProvider}
   * @param workspace the runtime's workspace
   * @param eventHandler the error eventHandler
   * @param symlinkPrefix the symlink prefix which should be removed
   * @param productName the product name
   */
  public static void removeOutputDirectoryLinks(
      Iterable<SymlinkDefinition> symlinkDefinitions,
      String workspaceName,
      Path workspace,
      EventHandler eventHandler,
      String symlinkPrefix,
      String productName) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }
    List<String> failures = new ArrayList<>();

    String workspaceBaseName = workspace.getBaseName();
    for (SymlinkDefinition link : getAllLinkDefinitions(symlinkDefinitions)) {
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
