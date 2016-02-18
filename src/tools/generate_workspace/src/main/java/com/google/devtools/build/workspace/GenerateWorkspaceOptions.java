// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.workspace;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import java.util.List;

/**
 * Command-line options for generate_workspace tool.
 */
public class GenerateWorkspaceOptions extends OptionsBase {
  @Option(
      name = "help",
      abbrev = 'h',
      help = "Prints usage info.",
      defaultValue = "true"
  )
  public boolean help;

  @Option(
      name = "bazel_project",
      abbrev = 'b',
      help = "Directory contains a Bazel project.",
      allowMultiple = true,
      category = "input",
      defaultValue = ""
  )
  public List<String> bazelProjects;

  @Option(
      name = "maven_project",
      abbrev = 'm',
      help = "Directory containing a Maven project.",
      allowMultiple = true,
      category = "input",
      defaultValue = ""
  )
  public List<String> mavenProjects;

  @Option(
      name = "artifact",
      abbrev = 'a',
      help = "Maven artifact coordinates (e.g. groupId:artifactId:version).",
      allowMultiple = true,
      category = "input",
      defaultValue = ""
  )
  public List<String> artifacts;
  
  @Option(
      name = "output_dir",
      abbrev = 'o',
      help = "Output directory to store the WORKSPACE and BUILD files. If unspecified, a temporary"
          + " directory is used.",
      category = "output",
      defaultValue = ""
  )
  public String outputDir;

}
