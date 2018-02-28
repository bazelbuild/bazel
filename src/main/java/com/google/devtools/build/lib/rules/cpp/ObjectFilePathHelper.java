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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.FileSystemUtils;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A helper Class for calculating the output names for object file paths from a set of source files.
 *
 * <p>The object file path is constructed in the following format:
 *    <bazel-bin>/<target_package_path>/_objs/<target_name>/<output_name>.<obj_extension>
 * When there's no two source files having the same basename:
 *    <output_name> = <source_file_base_name>
 * otherwise:
 *    <output_name> = N/<source_file_base_name>,
 *    N = the file’s order among the source files with the same basename
 *
 * Examples:
 *   1. Output names for ["lib1/foo.cc", "lib2/bar.cc"] are ["foo", "bar"]
 *   2. Output names for ["foo.cc", "bar.cc", "foo.cpp", "lib/foo.cc"] are
 *      ["0/foo", "bar", "1/foo", "2/foo"]
 */
public class ObjectFilePathHelper {

  private Map<Artifact, String> outputNameMap = new LinkedHashMap<>();

  public ObjectFilePathHelper(Iterable<Artifact> sourceArtifacts) {
    HashMap<String, Integer> count = new LinkedHashMap<>();
    HashMap<String, Integer> number = new LinkedHashMap<>();
    for (Artifact source : sourceArtifacts) {
      String outputName =
          FileSystemUtils.removeExtension(source.getRootRelativePath()).getBaseName();
      count.put(outputName, count.getOrDefault(outputName, 0) + 1);
    }

    for (Artifact source : sourceArtifacts) {
      String outputName =
          FileSystemUtils.removeExtension(source.getRootRelativePath()).getBaseName();
      if (count.getOrDefault(outputName, 0) > 1) {
        int num = number.getOrDefault(outputName, 0);
        number.put(outputName, num + 1);
        outputName = String.valueOf(num) + "/" + outputName;
      }
      outputNameMap.put(source, outputName);
    }
  }

  /**
   * Return the output name for the object file path of a given source file.
   */
  public String getOutputName(Artifact source) {
    return outputNameMap.get(source);
  }
}
