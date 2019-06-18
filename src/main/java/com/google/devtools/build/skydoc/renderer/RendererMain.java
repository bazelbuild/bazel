// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skydoc.renderer;

import com.google.devtools.common.options.OptionsParser;

/**
 * Main entry point for Renderer binary.
 *
 * <p>This Renderer will take in raw stardoc_proto protos as input and produce rich markdown output.
 */
public class RendererMain {
  // TODO(kendalllane, blossomsm): Implement.

  public static void main(String[] args) {
    OptionsParser parser = OptionsParser.newOptionsParser(RendererOptions.class);
    parser.parseAndExitUponError(args);
    RendererOptions rendererOptions = parser.getOptions(RendererOptions.class);

    if (rendererOptions.inputPath.isEmpty() || rendererOptions.outputFilePath.isEmpty()) {
      throw new IllegalArgumentException(
          "Both --input and --output must be specified. Usage: "
              + "{renderer_bin} --input=\"{input_proto_file}\" --output=\"{output_file}\"");
    }
  }
}
