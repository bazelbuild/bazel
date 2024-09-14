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

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.devtools.build.android.Converters.CompatExistingPathConverter;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import java.nio.file.Path;
import java.util.List;

/** The options for a {@link IdlClass} action. */
@Parameters(separators = "= ")
public final class IdlClassOptions {
  @Parameter(
      names = "--manifest_proto",
      converter = CompatExistingPathConverter.class,
      description = "The path to the manifest file output by the Java compiler.")
  public Path manifestProto;

  @Parameter(
      names = "--class_jar",
      converter = CompatExistingPathConverter.class,
      description = "The path to the class jar output by the Java compiler.")
  public Path classJar;

  @Parameter(
      names = "--output_class_jar",
      converter = CompatPathConverter.class,
      description = "The path to write the class jar output to.")
  public Path outputClassJar;

  @Parameter(
      names = "--output_source_jar",
      converter = CompatPathConverter.class,
      description = "The path to write the source jar output to.")
  public Path outputSourceJar;

  @Parameter(
      names = "--temp_dir",
      converter = CompatPathConverter.class,
      description = "The path to a temp directory.")
  public Path tempDir;

  @Parameter(
      names = "--idl_source_base_dir",
      converter = CompatPathConverter.class,
      description =
          "The path to the base directory of the idl sources. Optional; Used for testing.")
  public Path idlSourceBaseDir;

  @Parameter() public List<String> residue;
}
