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

import com.google.devtools.build.skydoc.rendering.MarkdownRenderer;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.UserDefinedFunctionInfo;
import com.google.devtools.common.options.OptionsParser;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * Main entry point for Renderer binary.
 *
 * <p>This Renderer takes in raw stardoc_proto protos as input and produces rich markdown output.
 */
public class RendererMain {

  public static void main(String[] args) throws IOException {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RendererOptions.class).build();
    parser.parseAndExitUponError(args);
    RendererOptions rendererOptions = parser.getOptions(RendererOptions.class);

    if (rendererOptions.inputPath.isEmpty() || rendererOptions.outputFilePath.isEmpty()) {
      throw new IllegalArgumentException(
          "Both --input and --output must be specified. Usage: "
              + "{renderer_bin} --input=\"{input_proto_file}\" --output=\"{output_file}\"");
    }

    String inputPath = rendererOptions.inputPath;
    String outputPath = rendererOptions.outputFilePath;
    MarkdownRenderer renderer = new MarkdownRenderer();
    try (PrintWriter printWriter = new PrintWriter(outputPath, "UTF-8")) {
      ModuleInfo moduleInfo = ModuleInfo.parseFrom(new FileInputStream(inputPath));
      printWriter.println(renderer.renderMarkdownHeader());
      printRuleInfos(printWriter, renderer, moduleInfo.getRuleInfoList());
      printProviderInfos(printWriter, renderer, moduleInfo.getProviderInfoList());
      printUserDefinedFunctions(printWriter, renderer, moduleInfo.getFuncInfoList());
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException("Input file is not a valid ModuleInfo proto.", e);
    }
  }

  private static void printRuleInfos(
      PrintWriter printWriter, MarkdownRenderer renderer, List<RuleInfo> ruleInfos)
      throws IOException {
    for (RuleInfo ruleProto : ruleInfos) {
      printWriter.println(renderer.render(ruleProto.getRuleName(), ruleProto));
    }
  }

  private static void printProviderInfos(
      PrintWriter printWriter, MarkdownRenderer renderer, List<ProviderInfo> providerInfos)
      throws IOException {
    for (ProviderInfo providerProto : providerInfos) {
      printWriter.println(renderer.render(providerProto.getProviderName(), providerProto));
    }
  }

  private static void printUserDefinedFunctions(
      PrintWriter printWriter,
      MarkdownRenderer renderer,
      List<UserDefinedFunctionInfo> userDefinedFunctions)
      throws IOException {
    for (UserDefinedFunctionInfo funcProto : userDefinedFunctions) {
      printWriter.println(renderer.render(funcProto));
    }
  }
}
