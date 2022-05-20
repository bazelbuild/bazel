// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildeventstream;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/** Options used to configure the build event protocol. */
public class BuildEventProtocolOptions extends OptionsBase {

  @Option(
    name = "legacy_important_outputs",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Use this to suppress generation of the legacy important_outputs field in the "
        + "TargetComplete event."
  )
  public boolean legacyImportantOutputs;

  @Option(
    name = "experimental_build_event_upload_strategy",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Selects how to upload artifacts referenced in the build event protocol."
  )
  public String buildEventUploadStrategy;

  @Option(
      name = "experimental_stream_log_file_uploads",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Stream log file uploads directly to the remote storage rather than writing them to"
              + " disk.")
  public boolean streamingLogFileUploads;

  @Option(
      name = "experimental_build_event_expand_filesets",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "If true, expand Filesets in the BEP when presenting output files.")
  public boolean expandFilesets;

  @Option(
      name = "experimental_build_event_fully_resolve_fileset_symlinks",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, fully resolve relative Fileset symlinks in the BEP when presenting output"
              + " files. Requires --experimental_build_event_expand_filesets.")
  public boolean fullyResolveFilesetSymlinks;

  @Option(
      name = "experimental_bep_target_summary",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Whether to publish TargetSummary events.")
  public boolean publishTargetSummary;
}
