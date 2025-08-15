package com.google.devtools.build.lib.remote.options;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

public final class RemoteStartupOptions extends OptionsBase {
  @Option(
      name = "experimental_remote_repo_contents_cache",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.REMOTE,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          """
          If enabled, the remote cache will be used to store the results of reproducible repository
          rules. If a repository rule needs to be evaluated and its result is already in the remote
          cache, the contents of the repository will be kept in an in-memory file system and are
          only downloaded when needed, either by Bazel itself or an action that runs locally.
          """)
  public boolean useRemoteRepoContentsCache;
}
