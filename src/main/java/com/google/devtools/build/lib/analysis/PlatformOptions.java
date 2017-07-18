// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelListConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;

/** Command-line options for platform-related configuration. */
public class PlatformOptions extends FragmentOptions {

  @Option(
    name = "experimental_host_platform",
    converter = BuildConfiguration.LabelConverter.class,
    defaultValue = "@bazel_tools//platforms:host_platform",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "Declare the platform the build is started from"
  )
  public Label hostPlatform;

  // TODO(katre): Add execution platforms.

  @Option(
    name = "experimental_platforms",
    converter = BuildConfiguration.LabelListConverter.class,
    defaultValue = "@bazel_tools//platforms:target_platform",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "Declare the platforms targeted by the current build"
  )
  public List<Label> platforms;

  @Option(
    name = "extra_toolchains",
    converter = LabelListConverter.class,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "Extra toolchains to be considered during toolchain resolution."
  )
  public List<Label> extraToolchains;

  @Override
  public PlatformOptions getHost(boolean fallback) {
    PlatformOptions host = (PlatformOptions) getDefault();
    host.platforms = ImmutableList.of(this.hostPlatform);
    return host;
  }
}
