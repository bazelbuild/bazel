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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

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
   */
  static SymlinkCreationResult createOutputDirectoryLinks(
      Iterable<SymlinkDefinition> symlinkDefinitions,
      BuildRequestOptions buildRequestOptions,
      String workspaceName,
      Path workspace,
      BlazeDirectories directories,
      EventHandler eventHandler,
      Set<BuildConfigurationValue> targetConfigs,
      String productName) {
    Path execRoot = directories.getExecRoot(workspaceName);
    Path outputPath = directories.getOutputPath(workspaceName);
    String symlinkPrefix = buildRequestOptions.getSymlinkPrefix(productName);
    ConvenienceSymlinksMode mode = buildRequestOptions.experimentalConvenienceSymlinks;
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return EMPTY_SYMLINK_CREATION_RESULT;
    }

    ImmutableList.Builder<ConvenienceSymlink> convenienceSymlinksBuilder = ImmutableList.builder();
    ImmutableMap.Builder<PathFragment, PathFragment> createdConvenienceSymlinksBuilder =
        ImmutableMap.builder();
    List<String> failures = new ArrayList<>();
    List<String> ambiguousLinks = new ArrayList<>();
    Set<String> createdLinks = new LinkedHashSet<>();
    String workspaceBaseName = workspace.getBaseName();
    RepositoryName repositoryName = RepositoryName.MAIN;
    boolean logOnly = mode == ConvenienceSymlinksMode.LOG_ONLY;

    for (SymlinkDefinition symlink : getAllLinkDefinitions(symlinkDefinitions)) {
      String linkName = symlink.getLinkName(symlinkPrefix, workspaceBaseName);
      if (!createdLinks.add(linkName)) {
        // already created a link by this name
        continue;
      }
      if (mode == ConvenienceSymlinksMode.CLEAN) {
        removeLink(workspace, linkName, failures, convenienceSymlinksBuilder, logOnly);
      } else {
        Set<Path> candidatePaths =
            symlink.getLinkPaths(
                buildRequestOptions, targetConfigs, repositoryName, outputPath, execRoot);
        if (candidatePaths.size() == 1) {
          createLink(
              workspace,
              linkName,
              execRoot,
              directories,
              Iterables.getOnlyElement(candidatePaths),
              failures,
              convenienceSymlinksBuilder,
              createdConvenienceSymlinksBuilder,
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
    return new SymlinkCreationResult(
        convenienceSymlinksBuilder.build(), createdConvenienceSymlinksBuilder.buildKeepingLast());
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
   */
  public static void removeOutputDirectoryLinks(
      Iterable<SymlinkDefinition> symlinkDefinitions,
      Path workspace,
      EventHandler eventHandler,
      String symlinkPrefix) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }
    List<String> failures = new ArrayList<>();

    String workspaceBaseName = workspace.getBaseName();

    for (SymlinkDefinition link : getAllLinkDefinitions(symlinkDefinitions)) {
      removeLink(
          workspace,
          link.getLinkName(symlinkPrefix, workspaceBaseName),
          failures,
          ImmutableList.builder(),
          false);
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
   * symlink. {@code execRoot} and {@code directories} are used to determine the relative target
   * path for this entry.
   *
   * <p>If {@code logOnly} is true, the {@code ConvenienceSymlink} entry is added but no actual
   * filesystem operations are performed.
   */
  private static void createLink(
      Path base,
      String name,
      Path execRoot,
      BlazeDirectories directories,
      Path target,
      List<String> failures,
      ImmutableList.Builder<ConvenienceSymlink> symlinksBuilder,
      ImmutableMap.Builder<PathFragment, PathFragment> createdSymlinksBuilder,
      boolean logOnly) {
    // The BEP event needs to report a target path relative to the output base. Usually the target
    // is already under the output base, but if the execroot is virtual (only happens in internal
    // blaze, see ModuleFileSystem), we need to rewrite the path using the real execroot.
    Path outputBase = directories.getOutputBase();
    Path targetForEvent =
        target.startsWith(outputBase)
            ? target
            : directories.getBlazeExecRoot().getRelative(target.relativeTo(execRoot));
    symlinksBuilder.add(
        ConvenienceSymlink.newBuilder()
            .setPath(name)
            .setTarget(targetForEvent.relativeTo(outputBase).getPathString())
            .setAction(Action.CREATE)
            .build());

    PathFragment nameFragment = PathFragment.create(name);
    if (logOnly) {
      // Still report as created - log-only implies we want to pretend it exists.
      createdSymlinksBuilder.put(nameFragment, target.asFragment());
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
      createdSymlinksBuilder.put(nameFragment, target.asFragment());
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
                RepositoryName repositoryName,
                Path outputPath,
                Path execRoot) {
              if (buildRequestOptions.incompatibleSkipGenfilesSymlink) {
                return ImmutableSet.of();
              }
              return super.getLinkPaths(
                  buildRequestOptions, targetConfigs, repositoryName, outputPath, execRoot);
            }
          },
          // output directory (bazel-out)
          new SymlinkDefinition() {
            @Override
            public String getLinkName(String symlinkPrefix, String workspaceBaseName) {
              return symlinkPrefix + "out";
            }

            @Override
            public ImmutableSet<Path> getLinkPaths(
                BuildRequestOptions buildRequestOptions,
                Set<BuildConfigurationValue> targetConfigs,
                RepositoryName repositoryName,
                Path outputPath,
                Path execRoot) {
              return ImmutableSet.of(outputPath);
            }
          },
          // execroot
          new SymlinkDefinition() {
            @Override
            public String getLinkName(String symlinkPrefix, String workspaceBaseName) {
              return symlinkPrefix + workspaceBaseName;
            }

            @Override
            public ImmutableSet<Path> getLinkPaths(
                BuildRequestOptions buildRequestOptions,
                Set<BuildConfigurationValue> targetConfigs,
                RepositoryName repositoryName,
                Path outputPath,
                Path execRoot) {
              return ImmutableSet.of(execRoot);
            }
          });

  static final SymlinkCreationResult EMPTY_SYMLINK_CREATION_RESULT =
      new SymlinkCreationResult(ImmutableList.of(), ImmutableMap.of());

  /** Describes the outcome of symlink creation. */
  static class SymlinkCreationResult {
    private final ImmutableList<ConvenienceSymlink> convenienceSymlinkProtos;
    private final ImmutableMap<PathFragment, PathFragment> createdSymlinks;

    private SymlinkCreationResult(
        ImmutableList<ConvenienceSymlink> convenienceSymlinkProtos,
        ImmutableMap<PathFragment, PathFragment> createdSymlinks) {
      this.convenienceSymlinkProtos = convenienceSymlinkProtos;
      this.createdSymlinks = createdSymlinks;
    }

    /** Returns descriptions of what symlinks were created and destroyed. */
    ImmutableList<ConvenienceSymlink> getConvenienceSymlinkProtos() {
      return convenienceSymlinkProtos;
    }

    /**
     * Returns symlink name -> target mappings of symlinks that were actually created (or in the
     * case of {@link ConvenienceSymlinksMode#LOG_ONLY}, would have been created).
     */
    ImmutableMap<PathFragment, PathFragment> getCreatedSymlinks() {
      return createdSymlinks;
    }
  }
}
