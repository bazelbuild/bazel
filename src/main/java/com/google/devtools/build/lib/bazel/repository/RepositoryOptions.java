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
    name = "experimental_repository_cache",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    metadataTags = {OptionMetadataTag.EXPERIMENTAL},
    converter = OptionsUtils.PathFragmentConverter.class,
    help =
        "Specifies the cache location of the downloaded values obtained "
            + "during the fetching of external repositories."
  )
  public PathFragment experimentalRepositoryCache;

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
