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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.SymlinkDefinition;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.ConvenienceSymlink;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.ConvenienceSymlink.Action;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions.ConvenienceSymlinksMode;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

/** Static utilities for managing output directory symlinks. */
public final class OutputDirectoryLinksUtils {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

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
  private static ImmutableList<SymlinkDefinition> getAllLinkDefinitions(
      Iterable<SymlinkDefinition> symlinkDefinitions) {
    ImmutableList.Builder<SymlinkDefinition> builder = ImmutableList.builder();
    builder.addAll(STANDARD_LINK_DEFINITIONS);
    builder.addAll(symlinkDefinitions);
    return builder.build();
  }

  private static final String NO_CREATE_SYMLINKS_PREFIX = "/";

  /**
   * Attempts to create or delete convenience symlinks in the workspace to the various output
   * directories, and generates associated log events.
   *
   * <p>If {@code --symlink_prefix} is {@link #NO_CREATE_SYMLINKS_PREFIX}, or {@code
   * --experimental_convenience_symlinks} is {@link ConvenienceSymlinksMode#IGNORE}, this method is
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
   * ConvenienceSymlinksMode#CLEAN}, all symlinks are set to be deleted. If it's set to {@link
   * ConvenienceSymlinksMode#NORMAL}, each symlink type decides whether it should be created or
   * deleted. (A symlink may decide to be deleted if e.g. it is disabled by a flag, or would want to
   * point to more than one destination.) If it's set to {@link ConvenienceSymlinksMode#LOG_ONLY},
   * the same logic is run as in the {@code NORMAL} case, but the result is only emitting log
   * messages, with no actual filesystem mutations.
   *
   * <p>A warning is emitted if a symlink would resolve to multiple destinations, or if a filesystem
   * mutation operation fails.
   *
   * @return a list of {@link ConvenienceSymlink} messages describing what was created and
   *     destroyed.
   */
  static ImmutableList<ConvenienceSymlink> createOutputDirectoryLinks(
      Iterable<SymlinkDefinition> symlinkDefinitions,
      BuildRequestOptions buildRequestOptions,
      String workspaceName,
      Path workspace,
      BlazeDirectories directories,
      EventHandler eventHandler,
      Set<BuildConfigurationValue> targetConfigs,
      Function<BuildOptions, BuildConfigurationValue> configGetter,
      String productName) {
    Path execRoot = directories.getExecRoot(workspaceName);
    Path outputPath = directories.getOutputPath(workspaceName);
    Path outputBase = directories.getOutputBase();
    String symlinkPrefix = buildRequestOptions.getSymlinkPrefix(productName);
    ConvenienceSymlinksMode mode = buildRequestOptions.experimentalConvenienceSymlinks;
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return ImmutableList.of();
    }

    ImmutableList.Builder<ConvenienceSymlink> convenienceSymlinksBuilder = ImmutableList.builder();
    List<String> failures = new ArrayList<>();
    List<String> ambiguousLinks = new ArrayList<>();
    Set<String> createdLinks = new LinkedHashSet<>();
    String workspaceBaseName = workspace.getBaseName();
    RepositoryName repositoryName = RepositoryName.MAIN;
    boolean logOnly = mode == ConvenienceSymlinksMode.LOG_ONLY;

    for (SymlinkDefinition symlink : getAllLinkDefinitions(symlinkDefinitions)) {
      String linkName = symlink.getLinkName(symlinkPrefix, productName, workspaceBaseName);
      if (!createdLinks.add(linkName)) {
        // already created a link by this name
        continue;
      }
      if (mode == ConvenienceSymlinksMode.CLEAN) {
        removeLink(workspace, linkName, failures, convenienceSymlinksBuilder, logOnly);
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
          createLink(
              workspace,
              linkName,
              outputBase,
              Iterables.getOnlyElement(candidatePaths),
              failures,
              convenienceSymlinksBuilder,
              logOnly);
        } else {
          removeLink(workspace, linkName, failures, convenienceSymlinksBuilder, logOnly);
          // candidatePaths can be empty if the symlink decided not to be created. This can happen
          // if the symlink is disabled by a flag, or it intercepts an error while computing its
          // target path. In that case, don't trigger a warning about an ambiguous link.
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
                  "cleared convenience symlink(s) %s because they wouldn't contain "
                      + "requested targets' outputs. Those targets self-transition to multiple "
                      + "distinct configurations",
                  Joiner.on(", ").join(ambiguousLinks))));
    }
    return convenienceSymlinksBuilder.build();
  }

  public static PathPrettyPrinter getPathPrettyPrinter(
      Iterable<SymlinkDefinition> symlinkDefinitions,
      String symlinkPrefix,
      String productName,
      Path workspaceDirectory) {
    return new PathPrettyPrinter(
        getAllLinkDefinitions(symlinkDefinitions), symlinkPrefix, productName, workspaceDirectory);
  }

  private static void removeAllSymlinks(
      List<String> failures, Path workspace, Path outputBase, String symlinkPrefix) {

    // Get the prefix directory relative to the workspace.
    Path symlinkPrefixDirectory = workspace.getRelative(symlinkPrefix);
    Path directoryWithSymlinks =
        // Handle the special case when the prefix is just a path.
        // Otherwise we need to go one level up.
        symlinkPrefix.endsWith("/")
            ? symlinkPrefixDirectory
            // This will return the workspace if the prefix was not a directory.
            : symlinkPrefixDirectory.getParentDirectory();
    Preconditions.checkNotNull(
        directoryWithSymlinks, "Invalid symlink_prefix provided: %s", symlinkPrefix);

    // Try to get a list of all symlinks in the workspace.
    Collection<Path> pathsInWorkspace;
    try {
      pathsInWorkspace = directoryWithSymlinks.getDirectoryEntries();
    } catch (IOException e) {
      failures.add(
          String.format(
              "Failed to list files under path %s: %s",
              directoryWithSymlinks.getPathString(), e.getMessage()));
      return;
    }

    // Iterate through all the files and delete any symbolic links that start with the prefix
    // and point to the output directory.
    for (Path entry : pathsInWorkspace) {
      try {
        if (entry.isSymbolicLink()
            && entry.relativeTo(workspace).getPathString().startsWith(symlinkPrefix)
            && entry.readSymbolicLink().startsWith(outputBase.asFragment())) {
          logger.atFinest().log("Removing %s", entry);
          entry.delete();
        }
      } catch (IOException e) {
        failures.add(String.format("%s: %s", entry.getBaseName(), e.getMessage()));
      }
    }
  }

  /**
   * Attempts to remove the convenience symlinks in the workspace directory.
   *
   * <p>Issues a warning if it fails, e.g. because workspaceDirectory is readonly. Also cleans up
   * any child directories created by a custom prefix.
   *
   * @param symlinkDefinitions extra symlink types added by the {@link ConfiguredRuleClassProvider}
   * @param workspace the runtime's workspace
   * @param outputBase the runtime's output directory. Only used with
   *     remove_all_convenience_symlinks.
   * @param eventHandler the error eventHandler
   * @param symlinkPrefix the symlink prefix which should be removed
   * @param productName the product name
   * @param removeAllConvenienceSymlinks Delete all symlinks with the given prefix, not just
   *     predefined links.
   */
  public static void removeOutputDirectoryLinks(
      Iterable<SymlinkDefinition> symlinkDefinitions,
      Path workspace,
      Path outputBase,
      EventHandler eventHandler,
      String symlinkPrefix,
      String productName,
      boolean removeAllConvenienceSymlinks) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }
    List<String> failures = new ArrayList<>();

    String workspaceBaseName = workspace.getBaseName();

    if (removeAllConvenienceSymlinks) {
      removeAllSymlinks(failures, workspace, outputBase, symlinkPrefix);
    } else {
      for (SymlinkDefinition link : getAllLinkDefinitions(symlinkDefinitions)) {
        removeLink(
            workspace,
            link.getLinkName(symlinkPrefix, productName, workspaceBaseName),
            failures,
            ImmutableList.builder(),
            false);
      }
    }

    FileSystemUtils.removeDirectoryAndParents(workspace, PathFragment.create(symlinkPrefix));
    if (!failures.isEmpty()) {
      eventHandler.handle(Event.warn(String.format(
          "failed to remove one or more convenience symlinks for prefix '%s':\n  %s", symlinkPrefix,
          Joiner.on("\n  ").join(failures))));
    }
  }

  /**
   * Creates a symlink and outputs a {@link ConvenienceSymlink} entry.
   *
   * <p>The symlink is created at path {@code name}, relative to {@code base}, creating directories
   * as needed; it points to {@code target}. Any filesystem errors are appended to {@code failures}.
   *
   * <p>A {@code ConvenienceSymlink} entry is added to {@code symlinksBuilder} describing the
   * symlink. {@code outputBase} is used to determine the relative target path for this entry.
   *
   * <p>If {@code logOnly} is true, the {@code ConvenienceSymlink} entry is added but no actual
   * filesystem operations are performed.
   */
  private static void createLink(
      Path base,
      String name,
      Path outputBase,
      Path target,
      List<String> failures,
      ImmutableList.Builder<ConvenienceSymlink> symlinksBuilder,
      boolean logOnly) {
    // Usually the symlink target falls under the output base, and the path in the BEP event should
    // be relative to that output base. In rare cases where the symlink points elsewhere, use the
    // absolute path as a fallback.
    String targetForEvent =
        target.startsWith(outputBase)
            ? target.relativeTo(outputBase).getPathString()
            : target.getPathString();
    symlinksBuilder.add(
        ConvenienceSymlink.newBuilder()
            .setPath(name)
            .setTarget(targetForEvent)
            .setAction(Action.CREATE)
            .build());
    if (logOnly) {
      return;
    }
    Path link = base.getRelative(name);
    try {
      target.createDirectoryAndParents();
    } catch (IOException e) {
      failures.add(String.format("cannot create directory %s: %s",
          target.getPathString(), e.getMessage()));
      return;
    }
    try {
      FileSystemUtils.ensureSymbolicLink(link, target);
    } catch (IOException e) {
      failures.add(String.format("cannot create symbolic link %s -> %s:  %s",
          name, target.getPathString(), e.getMessage()));
    }
  }

  /**
   * Deletes a symlink and outputs a {@link ConvenienceSymlink} entry.
   *
   * <p>The symlink to be deleted is at path {@code name}, relative to {@code base}. Any filesystem
   * errors are appended to {@code failures}.
   *
   * <p>A {@code ConvenienceSymlink} entry is added to {@code symlinksBuilder} describing the
   * symlink to be deleted.
   *
   * <p>If {@code logOnly} is true, the {@code ConvenienceSymlink} entry is added but no actual
   * filesystem operations are performed.
   */
  private static void removeLink(
      Path base,
      String name,
      List<String> failures,
      ImmutableList.Builder<ConvenienceSymlink> symlinksBuilder,
      boolean logOnly) {
    symlinksBuilder.add(
        ConvenienceSymlink.newBuilder().setPath(name).setAction(Action.DELETE).build());
    if (logOnly) {
      return;
    }
    Path link = base.getRelative(name);
    try {
      if (link.isSymbolicLink()) {
        // TODO(b/146885821): Consider also removing empty ancestor directories, to allow for
        //  cleaning up directories generated by --symlink_prefix=dir1/dir2/...
        //  Might be undesireable since it could also remove manually-created directories.
        logger.atFinest().log("Removing %s", link);
        link.delete();
      }
    } catch (IOException e) {
      failures.add(String.format("%s: %s", name, e.getMessage()));
    }
  }

  @SuppressWarnings("deprecation") // RuleContext#get*Directory not available here.
  private static final ImmutableList<SymlinkDefinition> STANDARD_LINK_DEFINITIONS =
      ImmutableList.of(
          new ConfigSymlink("bin", BuildConfigurationValue::getBinDirectory),
          new ConfigSymlink("testlogs", BuildConfigurationValue::getTestLogsDirectory),
          new ConfigSymlink("genfiles", BuildConfigurationValue::getGenfilesDirectory) {
            @Override
            public ImmutableSet<Path> getLinkPaths(
                BuildRequestOptions buildRequestOptions,
                Set<BuildConfigurationValue> targetConfigs,
                Function<BuildOptions, BuildConfigurationValue> configGetter,
                RepositoryName repositoryName,
                Path outputPath,
                Path execRoot) {
              if (buildRequestOptions.incompatibleSkipGenfilesSymlink) {
                return ImmutableSet.of();
              }
              return super.getLinkPaths(
                  buildRequestOptions,
                  targetConfigs,
                  configGetter,
                  repositoryName,
                  outputPath,
                  execRoot);
            }
          },
          // output directory (bazel-out)
          new SymlinkDefinition() {
            @Override
            public String getLinkName(
                String symlinkPrefix, String productName, String workspaceBaseName) {
              return symlinkPrefix + "out";
            }

            @Override
            public ImmutableSet<Path> getLinkPaths(
                BuildRequestOptions buildRequestOptions,
                Set<BuildConfigurationValue> targetConfigs,
                Function<BuildOptions, BuildConfigurationValue> configGetter,
                RepositoryName repositoryName,
                Path outputPath,
                Path execRoot) {
              return ImmutableSet.of(outputPath);
            }
          },
          // execroot
          new SymlinkDefinition() {
            @Override
            public String getLinkName(
                String symlinkPrefix, String productName, String workspaceBaseName) {
              return symlinkPrefix + workspaceBaseName;
            }

            @Override
            public ImmutableSet<Path> getLinkPaths(
                BuildRequestOptions buildRequestOptions,
                Set<BuildConfigurationValue> targetConfigs,
                Function<BuildOptions, BuildConfigurationValue> configGetter,
                RepositoryName repositoryName,
                Path outputPath,
                Path execRoot) {
              return ImmutableSet.of(execRoot);
            }
          });
}
