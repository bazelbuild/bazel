// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.aquery;

import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;

/** Options class for aquery specific query options. */
public class AqueryOptions extends CommonQueryOptions {
  @Option(
      name = "output",
      defaultValue = "text",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "The format in which the aquery results should be printed. Allowed values for aquery "
              + "are: text, textproto, proto, streamed_proto, jsonproto.")
  public String outputFormat;

  @Option(
      name = "include_commandline",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help = "Includes the content of the action command lines in the output (potentially large).")
  public boolean includeCommandline;

  @Option(
      name = "include_artifacts",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Includes names of the action inputs and outputs in the output " + "(potentially large).")
  public boolean includeArtifacts;

  @Option(
      name = "include_param_files",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Include the content of the param files used in the command (potentially large). "
              + "Note: Enabling this flag will automatically enable the "
              + "--include_commandline flag.")
  public boolean includeParamFiles;

  @Option(
      name = "include_file_write_contents",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Include the file contents for the FileWrite, SourceSymlinkManifest, and "
              + "RepoMappingManifest actions (potentially large). ")
  public boolean includeFileWriteContents;

  @Option(
      name = "skyframe_state",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Without performing extra analysis, dump the current Action Graph from Skyframe. "
              + "Note: Specifying a target with --skyframe_state is currently not supported. "
              + "This flag is only available with --output=proto or --output=textproto.")
  public boolean queryCurrentSkyframeState;
}
