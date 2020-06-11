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
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
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
     * <p>The symlink should only be created if there is exactly one candidate. Zero candidates is a
     * no-op, and more than one candidate means a warning about ambiguous symlink destinations
     * should be emitted.
     *
     * @param buildRequestOptions options that may control which symlinks get created and what they
     *     point to.
     * @param targetConfigs the configurations for which symlinks should be created. If these have
     *     conflicting requirements, multiple candidates are returned.
     * @param configGetter used to compute derived configurations, if needed. This is used for
     *     symlinks that link to the output directories of configs that are related to, but not
     *     included in, {@code targetConfigs}.
     * @param repositoryName the repository name.
     * @param outputPath the output path.
     * @param execRoot the exec root.
     */
    Set<Path> getLinkPaths(
        BuildRequestOptions buildRequestOptions,
        Set<BuildConfiguration> targetConfigs,
        Function<BuildOptions, BuildConfiguration> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot);
  }

  /** Base class for symlinks to output roots. */
  private static class ConfigSymlink implements SymlinkDefinition {
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
        BuildRequestOptions buildRequestOptions,
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

  public static final ConfigSymlink BIN_SYMLINK =
      new ConfigSymlink("bin", BuildConfiguration::getBinDirectory);

  public static final ConfigSymlink TESTLOGS_SYMLINK =
      new ConfigSymlink("testlogs", BuildConfiguration::getTestLogsDirectory);

  public static final ConfigSymlink GENFILES_SYMLINK =
      new ConfigSymlink("genfiles", BuildConfiguration::getGenfilesDirectory) {
        @Override
        public Set<Path> getLinkPaths(
            BuildRequestOptions buildRequestOptions,
            Set<BuildConfiguration> targetConfigs,
            Function<BuildOptions, BuildConfiguration> configGetter,
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
      };

  /** Symlink to the execroot. */
  public enum ExecRootSymlink implements SymlinkDefinition {
    INSTANCE;

    @Override
    public String getLinkName(String symlinkPrefix, String productName, String workspaceBaseName) {
      return symlinkPrefix + workspaceBaseName;
    }

    @Override
    public Set<Path> getLinkPaths(
        BuildRequestOptions buildRequestOptions,
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
    // TODO(mitchellhyang): Symlink is omitted if '--experimental_no_product_name_out_symlink=true'.
    //  After confirming that if '--symlink_prefix' is used and '<product>-out' symlink is no longer
    //  needed, PRODUCT_NAME can be removed.
    PRODUCT_NAME {
      @Override
      public String getLinkName(
          String symlinkPrefix, String productName, String workspaceBaseName) {
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
        BuildRequestOptions buildRequestOptions,
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
   */
  public static ImmutableList<SymlinkDefinition> getStandardLinkDefinitions(
      boolean includeProductOut) {
    ImmutableList.Builder<SymlinkDefinition> builder = ImmutableList.builder();
    builder.add(BIN_SYMLINK);
    builder.add(TESTLOGS_SYMLINK);
    builder.add(GENFILES_SYMLINK);
    if (includeProductOut) {
      builder.add(OutputSymlink.PRODUCT_NAME);
    }
    builder.add(OutputSymlink.SYMLINK_PREFIX);
    builder.add(ExecRootSymlink.INSTANCE);
    return builder.build();
  }
}
