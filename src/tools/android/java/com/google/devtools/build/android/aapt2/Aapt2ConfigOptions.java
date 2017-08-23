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
// Copyright 2017 The Bazel Authors. All rights reserved.

package com.google.devtools.build.android.aapt2;

import com.android.repository.Revision;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.RevisionConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.nio.file.Path;

/** Aaprt2 specific configuration options. */
public class Aapt2ConfigOptions extends OptionsBase {
  @Option(
    name = "aapt2",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    defaultValue = "null",
    converter = ExistingPathConverter.class,
    category = "tool",
    help = "Aapt2 tool location for resource compilation."
  )
  public Path aapt2;

  @Option(
    name = "buildToolsVersion",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    defaultValue = "null",
    converter = RevisionConverter.class,
    category = "config",
    help = "Version of the build tools (e.g. aapt) being used, e.g. 23.0.2"
  )
  public Revision buildToolsVersion;

  @Option(
    name = "androidJar",
    defaultValue = "null",
    converter = ExistingPathConverter.class,
    category = "tool",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Path to the android jar for resource packaging and building apks."
  )
  public Path androidJar;
}
