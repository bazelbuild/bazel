// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.idlclass;

import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.nio.file.Path;

/** The options for a {@IdlClass} action. */
public final class IdlClassOptions extends OptionsBase {
  @Option(
    name = "manifest_proto",
    defaultValue = "null",
    converter = ExistingPathConverter.class,
    category = "input",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The path to the manifest file output by the Java compiler."
  )
  public Path manifestProto;

  @Option(
    name = "class_jar",
    defaultValue = "null",
    converter = ExistingPathConverter.class,
    category = "input",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The path to the class jar output by the Java compiler."
  )
  public Path classJar;

  @Option(
    name = "output_class_jar",
    defaultValue = "null",
    converter = PathConverter.class,
    category = "output",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The path to write the class jar output to."
  )
  public Path outputClassJar;

  @Option(
    name = "output_source_jar",
    defaultValue = "null",
    converter = PathConverter.class,
    category = "output",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The path to write the source jar output to."
  )
  public Path outputSourceJar;

  @Option(
    name = "temp_dir",
    defaultValue = "null",
    converter = PathConverter.class,
    category = "input",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The path to a temp directory."
  )
  public Path tempDir;

  @Option(
    name = "idl_source_base_dir",
    defaultValue = "null",
    converter = PathConverter.class,
    category = "input",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The path to the base directory of the idl sources. Optional; Used for testing."
  )
  public Path idlSourceBaseDir;
}
