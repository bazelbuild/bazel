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
package com.google.devtools.common.options.processor.optiontestsources;

import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

/**
 * This example options class checks multiple combinations of list-type options that should all
 * successfully compile.
 */
public class MultipleOptionWithListTypeConverter extends OptionsBase {
  @Option(
      name = "multiple_strings_multiple_times_grouped",
      defaultValue = "null",
      converter = CommaSeparatedOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      allowMultiple = true
  )
  public List<List<String>> multipleStringsKeptInGroups;

  @Option(
      name = "multiple_strings_multiple_times_concatenated",
      defaultValue = "null",
      converter = CommaSeparatedOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      allowMultiple = true
  )
  public List<String> multipleStringsConcatenated; // Not List<List<String>>

  @Option(
    name = "multiple_strings_single_time",
    defaultValue = "a,b,c",
    converter = CommaSeparatedOptionListConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.NO_OP}
  )
  public List<String> multipleStringsSingleMention;
}
