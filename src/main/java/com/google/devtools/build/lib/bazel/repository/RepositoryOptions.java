// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.DurationConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.time.Duration;
import java.util.List;
import net.starlark.java.eval.EvalException;

/** Command-line options for repositories. */
public class RepositoryOptions extends OptionsBase {

  @Option(
      name = "repository_cache",
      oldName = "experimental_repository_cache",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          """
          Specifies the cache location of the downloaded values obtained
          during the fetching of external repositories. An empty string
          as argument requests the cache to be disabled,
          otherwise the default of `<--output_user_root>/cache/repos/v1` is used.
          """)
  public PathFragment repositoryCache;

  @Option(
      name = "repo_contents_cache",
      oldName = "repository_contents_cache",
      oldNameWarning = false,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          """
          Specifies the location of the repo contents cache, which contains fetched repo
          directories shareable across workspaces. An empty string as argument requests the repo
          contents cache to be disabled, otherwise the default of `<--repository_cache>/contents`
          is used. Note that this means setting `--repository_cache=` would by default disable the
          repo contents cache as well, unless `--repo_contents_cache=<some_path>` is also set.
          """)
  public PathFragment repoContentsCache;

  @Option(
      name = "repo_contents_cache_gc_max_age",
      defaultValue = "14d",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = DurationConverter.class,
      help =
          """
          Specifies the amount of time an entry in the repo contents cache can stay unused before
          it's garbage collected. If set to zero, garbage collection is disabled.
          """)
  public Duration repoContentsCacheGcMaxAge;

  @Option(
      name = "repo_contents_cache_gc_idle_delay",
      defaultValue = "5m",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = DurationConverter.class,
      help =
          """
          Specifies the amount of time the server must remain idle before garbage collection happens
          to the repo contents cache.
          """)
  public Duration repoContentsCacheGcIdleDelay;

  @Option(
      name = "registry",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "Specifies the registries to use to locate Bazel module dependencies. The order is"
              + " important: modules will be looked up in earlier registries first, and only fall"
              + " back to later registries when they're missing from the earlier ones.")
  public List<String> registries;

  @Option(
      name = "allow_yanked_versions",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          """
          Specified the module versions in the form of
          `<module1>@<version1>,<module2>@<version2>` that will be allowed in the resolved
          dependency graph even if they are declared yanked in the registry where they come
          from (if they are not coming from a [`NonRegistryOverride`]). Otherwise, yanked
          versions will cause the resolution to fail. You can also define allowed yanked
          version with the `BZLMOD_ALLOW_YANKED_VERSIONS` environment variable. You can
          disable this check by using the keyword `all` (not recommended).
          [`NonRegistryOverride`]: https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/bazel/bzlmod/NonRegistryOverride.java
          """)
  public List<String> allowedYankedVersions;

  @Option(
      name = "experimental_repository_cache_hardlinks",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "If set, the repository cache will hardlink the file in case of a"
              + " cache hit, rather than copying. This is intended to save disk space.")
  public boolean useHardlinks;

  @Option(
      name = "repository_disable_download",
      oldName = "experimental_repository_disable_download",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          """
          If set, downloading using `ctx.download{,_and_extract}` is not allowed during repository
          fetching. Note that network access is not completely disabled; ctx.execute could
          still run an arbitrary executable that accesses the Internet.
          """)
  public boolean disableDownload;

  @Option(
      name = "experimental_repository_downloader_retries",
      defaultValue = "5",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "The maximum number of attempts to retry a download error. If set to 0, retries are"
              + " disabled.")
  public int repositoryDownloaderRetries;

  @Option(
      name = "distdir",
      oldName = "experimental_distdir",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "Additional places to search for archives before accessing the network "
              + "to download them.")
  public List<PathFragment> experimentalDistdir;

  @Option(
      name = "override_repository",
      defaultValue = "null",
      allowMultiple = true,
      converter = RepositoryOverrideConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          """
          Override a repository with a local path in the form of `<repository name>=<path>`. If the
          given path is an absolute path, it will be used as it is. If the given path is a
          relative path, it is relative to the current working directory. If the given path
          starts with `%workspace%`, it is relative to the workspace root, which is the
          output of `bazel info workspace`. If the given path is empty, then remove any
          previous overrides.
          """)
  public List<RepositoryOverride> repositoryOverrides;

  @Option(
      name = "inject_repository",
      defaultValue = "null",
      allowMultiple = true,
      converter = RepositoryInjectionConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          """
          Adds a new repository with a local path in the form of `<repository name>=<path>`. This
          only takes effect with `--enable_bzlmod` and is equivalent to adding a
          corresponding `local_repository` to the root module's `MODULE.bazel` file via
          `use_repo_rule`. If the given path is an absolute path, it will be used as it is.
          If the given path is a relative path, it is relative to the current working
          directory. If the given path starts with `%workspace%`, it is relative to the
          workspace root, which is the output of `bazel info workspace`. If the given path
          is empty, then remove any previous injections.
          """)
  public List<RepositoryInjection> repositoryInjections;

  @Option(
      name = "override_module",
      defaultValue = "null",
      allowMultiple = true,
      converter = ModuleOverrideConverter.class,
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          """
          Override a module with a local path in the form of `<module name>=<path>`. If the given
          path is an absolute path, it will be used as it is. If the given path is a
          relative path, it is relative to the current working directory. If the given path
          starts with `%workspace%`, it is relative to the workspace root, which is the
          output of `bazel info workspace`. If the given path is empty, then remove any
          previous overrides.
          """)
  public List<ModuleOverride> moduleOverrides;

  @Option(
      name = "experimental_scale_timeouts",
      defaultValue = "1.0",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "Scale all timeouts in Starlark repository rules by this factor."
              + " In this way, external repositories can be made working on machines"
              + " that are slower than the rule author expected, without changing the"
              + " source code")
  public double experimentalScaleTimeouts;

  @Option(
      name = "downloader_config",
      oldName = "experimental_downloader_config",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.REMOTE,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "Specify a file to configure the remote downloader with. This file consists of lines, "
              + "each of which starts with a directive (`allow`, `block` or `rewrite`) followed "
              + "by either a host name (for `allow` and `block`) or two patterns, one to match "
              + "against, and one to use as a substitute URL, with back-references starting from "
              + "`$1`. It is possible for multiple `rewrite` directives for the same URL to be "
              + "given, and in this case multiple URLs will be returned.")
  public PathFragment downloaderConfig;

  @Option(
      name = "ignore_dev_dependency",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          """
          If true, Bazel ignores `bazel_dep` and `use_extension` declared as `dev_dependency` in
          the `MODULE.bazel` of the root module. Note that, those dev dependencies are always
          ignored in the `MODULE.bazel` if it's not the root module regardless of the value
          of this flag.
          """)
  public boolean ignoreDevDependency;

  @Option(
      name = "check_direct_dependencies",
      defaultValue = "warning",
      converter = CheckDirectDepsMode.Converter.class,
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Check if the direct `bazel_dep` dependencies declared in the root module are the same"
              + " versions you get in the resolved dependency graph. Valid values are `off` to"
              + " disable the check, `warning` to print a warning when mismatch detected or `error`"
              + " to escalate it to a resolution failure.")
  public CheckDirectDepsMode checkDirectDependencies;

  @Option(
      name = "experimental_check_external_repository_files",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Check for modifications to files in external repositories. Consider setting "
              + "this flag to false if you don't expect these files to change outside of bazel "
              + "since it will speed up subsequent runs as they won't have to check a "
              + "previous run's cache.")
  public boolean checkExternalRepositoryFiles;

  @Option(
      name = "check_bazel_compatibility",
      defaultValue = "error",
      converter = BazelCompatibilityMode.Converter.class,
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Check bazel version compatibility of Bazel modules. Valid values are `error` to escalate"
              + " it to a resolution failure, `off` to disable the check, or `warning` to print a"
              + " warning when mismatch detected.")
  public BazelCompatibilityMode bazelCompatibilityMode;

  @Option(
      name = "lockfile_mode",
      converter = LockfileMode.Converter.class,
      defaultValue = "update",
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Specifies how and whether or not to use the lockfile. Valid values are `update` to"
              + " use the lockfile and update it if there are changes, `refresh` to additionally"
              + " refresh mutable information (yanked versions and previously missing modules)"
              + " from remote registries from time to time, `error` to use the lockfile but throw"
              + " an error if it's not up-to-date, or `off` to neither read from or write to the"
              + " lockfile.")
  public LockfileMode lockfileMode;

  @Option(
      name = "vendor_dir",
      defaultValue = "null",
      converter = OptionsUtils.PathFragmentConverter.class,
      documentationCategory = OptionDocumentationCategory.BZLMOD,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Specifies the directory that should hold the external repositories in vendor mode, "
              + "whether for the purpose of fetching them into it or using them while building. "
              + "The path can be specified as either an absolute path or a path relative to the "
              + "workspace directory.")
  public PathFragment vendorDirectory;

  /** An enum for specifying different modes for checking direct dependency accuracy. */
  public enum CheckDirectDepsMode {
    OFF, // Don't check direct dependency accuracy.
    WARNING, // Print warning when mismatch.
    ERROR; // Throw an error when mismatch.

    /** Converts to {@link CheckDirectDepsMode}. */
    public static class Converter extends EnumConverter<CheckDirectDepsMode> {
      public Converter() {
        super(CheckDirectDepsMode.class, "direct deps check mode");
      }
    }
  }

  /** An enum for specifying different modes for bazel compatibility check. */
  public enum BazelCompatibilityMode {
    ERROR, // Check and throw an error when mismatched.
    WARNING, // Print warning when mismatched.
    OFF; // Don't check bazel version compatibility.

    /** Converts to {@link BazelCompatibilityMode}. */
    public static class Converter extends EnumConverter<BazelCompatibilityMode> {
      public Converter() {
        super(BazelCompatibilityMode.class, "Bazel compatibility check mode");
      }
    }
  }

  /** An enum for specifying how to use the lockfile. */
  public enum LockfileMode {
    OFF, // Don't use the lockfile at all.
    UPDATE, // Update the lockfile wh
    REFRESH,
    ERROR; // Throw an error when it mismatc

    /** Converts to {@link LockfileMode}. */
    public static class Converter extends EnumConverter<LockfileMode> {
      public Converter() {
        super(LockfileMode.class, "Lockfile mode");
      }
    }
  }

  /**
   * Converts from an equals-separated pair of strings into RepositoryName->PathFragment mapping.
   */
  public static class RepositoryOverrideConverter
      extends Converter.Contextless<RepositoryOverride> {

    @Override
    public RepositoryOverride convert(String input) throws OptionsParsingException {
      String[] pieces = input.split("=", 2);
      if (pieces.length != 2) {
        throw new OptionsParsingException(
            "Repository overrides must be of the form 'repository-name=path'", input);
      }
      OptionsUtils.PathFragmentConverter pathConverter = new OptionsUtils.PathFragmentConverter();
      String pathString = pathConverter.convert(pieces[1]).getPathString();
      try {
        return new RepositoryOverride(RepositoryName.create(pieces[0]), pathString);
      } catch (LabelSyntaxException e) {
        throw new OptionsParsingException("Invalid repository name given to override", input, e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "an equals-separated mapping of repository name to path";
    }
  }

  /**
   * Converts from an equals-separated pair of strings into RepositoryName->PathFragment mapping.
   */
  public static class RepositoryInjectionConverter
      extends Converter.Contextless<RepositoryInjection> {

    @Override
    public RepositoryInjection convert(String input) throws OptionsParsingException {
      String[] pieces = input.split("=", 2);
      if (pieces.length != 2) {
        throw new OptionsParsingException(
            "Repository injections must be of the form 'repository-name=path'", input);
      }
      OptionsUtils.PathFragmentConverter pathConverter = new OptionsUtils.PathFragmentConverter();
      String pathString = pathConverter.convert(pieces[1]).getPathString();
      try {
        RepositoryName.validateUserProvidedRepoName(pieces[0]);
        return new RepositoryInjection(pieces[0], pathString);
      } catch (EvalException e) {
        throw new OptionsParsingException("Invalid repository name given to inject", input, e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "an equals-separated mapping of repository name to path";
    }
  }

  /** Converts from an equals-separated pair of strings into ModuleName->PathFragment mapping. */
  public static class ModuleOverrideConverter extends Converter.Contextless<ModuleOverride> {

    @Override
    public ModuleOverride convert(String input) throws OptionsParsingException {
      String[] pieces = input.split("=", 2);
      if (pieces.length != 2) {
        throw new OptionsParsingException(
            "Module overrides must be of the form 'module-name=path'", input);
      }

      if (!RepositoryName.VALID_MODULE_NAME.matcher(pieces[0]).matches()) {
        throw new OptionsParsingException(
            String.format(
                "invalid module name '%s': valid names must 1) only contain lowercase letters"
                    + " (a-z), digits (0-9), dots (.), hyphens (-), and underscores (_); 2) begin"
                    + " with a lowercase letter; 3) end with a lowercase letter or digit.",
                pieces[0]));
      }

      OptionsUtils.PathFragmentConverter pathConverter = new OptionsUtils.PathFragmentConverter();
      String pathString = pathConverter.convert(pieces[1]).getPathString();
      return new ModuleOverride(pieces[0], pathString);
    }

    @Override
    public String getTypeDescription() {
      return "an equals-separated mapping of module name to path";
    }
  }

  /** A repository override, represented by a name and an absolute path to a repository. */
  public record RepositoryOverride(RepositoryName repositoryName, String path) {}

  /**
   * A repository injected into the scope of the root module, represented by a name and an absolute
   * path to a repository.
   */
  public record RepositoryInjection(String apparentName, String path) {}

  /** A module override, represented by a name and an absolute path to a module. */
  public record ModuleOverride(String moduleName, String path) {}
}
