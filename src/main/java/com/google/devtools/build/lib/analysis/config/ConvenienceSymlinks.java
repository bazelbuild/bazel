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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Set;
import java.util.function.Function;

/**
 * A static utility class containing definitions for creating the convenience symlinks after the
 * build.
 *
 * <p>Convenience symlinks are the symlinks that appear in the workspace and point to output
 * directories, e.g. {@code bazel-bin}.
 *
 * <p>For the actual logic to create and manage these symlinks, see {@link
 * OutputDirectoryLinksUtils}.
 */
public final class ConvenienceSymlinks {

  // Static utility class.
  private ConvenienceSymlinks() {}

  /** Represents a single kind of convenience symlink ({@code bazel-bin}, etc.). */
  public interface SymlinkDefinition {
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
     * <p>The symlink should only be created if there is exactly one candidate. Zero candidates is
     * a no-op, and more than one candidate means a warning about ambiguous symlink destinations
     * should be emitted.
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

  /** Base class for symlinks to output roots. */
  public static final class ConfigSymlink implements SymlinkDefinition {
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

  /** Symlink to the execroot. */
  public enum ExecRootSymlink implements SymlinkDefinition {
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

  /** Symlinks to the output directory. */
  public enum OutputSymlink implements SymlinkDefinition {
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

  /**
   * Returns the standard types of convenience symlinks.
   *
   * <p>The order of the result indicates precedence for {@link PathPrettyPrinter}.
   *
   * @param includeGenfiles whether to include the {@code genfiles} symlink, which is in the process
   *     of being deprecated ({@code --incompatible_skip_genfiles_symlink})
   */
  public static ImmutableList<SymlinkDefinition> getStandardLinkDefinitions(
      boolean includeGenfiles) {
    ImmutableList.Builder<SymlinkDefinition> builder = ImmutableList.builder();
    builder.add(new ConfigSymlink("bin", BuildConfiguration::getBinDirectory));
    builder.add(new ConfigSymlink("testlogs", BuildConfiguration::getTestLogsDirectory));
    if (includeGenfiles) {
      builder.add(new ConfigSymlink("genfiles", BuildConfiguration::getGenfilesDirectory));
    }
    builder.add(OutputSymlink.PRODUCT_NAME);
    builder.add(OutputSymlink.SYMLINK_PREFIX);
    builder.add(ExecRootSymlink.INSTANCE);
    return builder.build();
  }
}
