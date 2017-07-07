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

package com.google.devtools.build.benchmark;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.proto.OptionFilters.OptionEffectTag;
import java.util.List;

/** Class that contains arguments for running the benchmark. */
public class BenchmarkOptions extends OptionsBase {

  @Option(
    name = "workspace",
    defaultValue = "",
    category = "benchmark",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    valueHelp = "path",
    help = "Directory where we put all the code and results."
  )
  public String workspace;

  @Option(
    name = "output",
    defaultValue = "",
    category = "benchmark",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    valueHelp = "path",
    help = "Path to put benchmark result (json format)."
  )
  public String output;

  @Option(
    name = "version_between",
    defaultValue = "",
    category = "version filter",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    valueHelp = "string",
    help = "Use code versions between two versions, eg. 'abcedf..uvwxyz'.",
    converter = VersionFilterConverter.class
  )
  public VersionFilter versionFilter;

  @Option(
    name = "time_between",
    defaultValue = "",
    category = "time filter",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    valueHelp = "string",
    help = "Use code versions between two time, eg. '2017-01-01 13:00..2017-01-02 08:00'.",
    converter = DateFilterConverter.class
  )
  public DateFilter dateFilter;

  @Option(
    name = "versions",
    defaultValue = "",
    category = "version",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    valueHelp = "list of strings",
    allowMultiple = true,
    help = "Use code versions as listed."
  )
  public List<String> versions;
}
