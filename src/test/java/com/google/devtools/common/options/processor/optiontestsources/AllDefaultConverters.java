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

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;
import java.time.Duration;

/**
 * This class should contain all of the types with DEFAULT_CONVERTERS, and each converter should be
 * found without generating compilation errors.
 */
public class AllDefaultConverters extends OptionsBase {
  @Option(
    name = "boolean_option",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public boolean booleanOption;

  @Option(
    name = "double_option",
    defaultValue = "42.73",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public double doubleOption;

  @Option(
    name = "int_option",
    defaultValue = "42",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public int intOption;

  @Option(
    name = "long_option",
    defaultValue = "-5000000000000",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public long longOption;

  @Option(
    name = "string_option",
    defaultValue = "strings are strings are strings are strings",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public String stringOption;

  @Option(
    name = "tri_state_option",
    defaultValue = "auto",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public TriState triStateOption;

  @Option(
    name = "duration_option",
    defaultValue = "3600s",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public Duration durationOption;

  @Option(
    name = "void_option",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS}
  )
  public Void voidOption;
}
