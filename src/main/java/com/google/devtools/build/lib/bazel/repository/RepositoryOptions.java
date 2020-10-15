//Copyright 2016 The Bazel Authors. All rights reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

package com.google.devtools.build.lib.bazel.repository;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;

/**
 * Command-line options for repositories.
 */
public class RepositoryOptions extends OptionsBase {

  @Option(
      name = "repository_cache",
      oldName = "experimental_repository_cache",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "Specifies the cache location of the downloaded values obtained "
              + "during the fetching of external repositories. An empty string "
              + "as argument requests the cache to be disabled.")
  public PathFragment experimentalRepositoryCache;

  @Option(
      name = "experimental_repository_cache_hardlinks",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "If set, the repository cache will hardlink the file in case of a"
              + " cache hit, rather than copying. This is inteded to save disk space.")
  public boolean useHardlinks;

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
      name = "http_timeout_scaling",
      defaultValue = "1.0",
      documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "Scale all timeouts related to http downloads by the given factor")
  public double httpTimeoutScaling;

  @Option(
    name = "override_repository",
    defaultValue = "null",
    allowMultiple = true,
    converter = RepositoryOverrideConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Overrides a repository with a local directory."
  )
  public List<RepositoryOverride> repositoryOverrides;

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
      name = "experimental_repository_hash_file",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If non-empty, specifies a file containing a resolved value, against which"
              + " the repository directory hashes should be verified")
  public String repositoryHashFile;

  @Option(
      name = "experimental_verify_repository_rules",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If list of repository rules for which the hash of the output directory should be"
              + " verified, provided a file is specified by"
              + " --experimental_repository_hash_file.")
  public List<String> experimentalVerifyRepositoryRules;

  @Option(
      name = "experimental_resolved_file_instead_of_workspace",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help = "If non-empty read the specified resolved file instead of the WORKSPACE file")
  public String experimentalResolvedFileInsteadOfWorkspace;

  @Option(
      name = "experimental_downloader_config",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.REMOTE,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify a file to configure the remote downloader with. This file consists of lines, "
              + "each of which starts with a directive (`allow`, `block` or `rewrite`) followed "
              + "by either a host name (for `allow` and `block`) or two patterns, one to match "
              + "against, and one to use as a substitute URL, with back-references starting from "
              + "`$1`. It is possible for multiple `rewrite` directives for the same URL to be "
              + "give, and in this case multiple URLs will be returned.")
  public String downloaderConfig;

  /**
   * Converts from an equals-separated pair of strings into RepositoryName->PathFragment mapping.
   */
  public static class RepositoryOverrideConverter implements Converter<RepositoryOverride> {

    @Override
    public RepositoryOverride convert(String input) throws OptionsParsingException {
      String[] pieces = input.split("=");
      if (pieces.length != 2) {
        throw new OptionsParsingException(
            "Repository overrides must be of the form 'repository-name=path'", input);
      }
      PathFragment path = PathFragment.create(pieces[1]);
      if (!path.isAbsolute()) {
        throw new OptionsParsingException(
            "Repository override directory must be an absolute path", input);
      }
      try {
        return RepositoryOverride.create(RepositoryName.create("@" + pieces[0]), path);
      } catch (LabelSyntaxException e) {
        throw new OptionsParsingException("Invalid repository name given to override", input);
      }
    }

    @Override
    public String getTypeDescription() {
      return "an equals-separated mapping of repository name to path";
    }
  }

  /**
   * A repository override, represented by a name and an absolute path to a repository.
   */
  @AutoValue
  public abstract static class RepositoryOverride {

    private static RepositoryOverride create(RepositoryName repositoryName, PathFragment path) {
      return new AutoValue_RepositoryOptions_RepositoryOverride(repositoryName, path);
    }

    public abstract RepositoryName repositoryName();
    public abstract PathFragment path();
  }
}
