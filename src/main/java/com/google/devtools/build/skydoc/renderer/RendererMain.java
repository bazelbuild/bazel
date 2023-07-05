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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.skydoc.rendering.MarkdownRenderer;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.devtools.common.options.OptionsParser;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
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

    if (rendererOptions.headerTemplateFilePath.isEmpty()
        || rendererOptions.ruleTemplateFilePath.isEmpty()
        || rendererOptions.providerTemplateFilePath.isEmpty()
        || rendererOptions.funcTemplateFilePath.isEmpty()
        || rendererOptions.aspectTemplateFilePath.isEmpty()) {
      throw new FileNotFoundException(
          "Input templates --header_template --func_template --provider_template --rule_template"
              + " --aspect_template must be specified.");
    }

    String headerTemplatePath = rendererOptions.headerTemplateFilePath;
    String ruleTemplatePath = rendererOptions.ruleTemplateFilePath;
    String providerTemplatePath = rendererOptions.providerTemplateFilePath;
    String funcTemplatePath = rendererOptions.funcTemplateFilePath;
    String aspectTemplatePath = rendererOptions.aspectTemplateFilePath;

    MarkdownRenderer renderer =
        new MarkdownRenderer(
            headerTemplatePath,
            ruleTemplatePath,
            providerTemplatePath,
            funcTemplatePath,
            aspectTemplatePath);
    try (PrintWriter printWriter =
        new PrintWriter(outputPath, UTF_8) {
          // Use consistent line endings on all platforms.
          @Override
          public void println() {
            write("\n");
          }
        }) {
      ModuleInfo moduleInfo = ModuleInfo.parseFrom(new FileInputStream(inputPath));
      printWriter.println(renderer.renderMarkdownHeader(moduleInfo));
      printRuleInfos(printWriter, renderer, moduleInfo.getRuleInfoList());
      printProviderInfos(printWriter, renderer, moduleInfo.getProviderInfoList());
      printStarlarkFunctions(printWriter, renderer, moduleInfo.getFuncInfoList());
      printAspectInfos(printWriter, renderer, moduleInfo.getAspectInfoList());
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException("Input file is not a valid ModuleInfo proto.", e);
    }
  }

  private static void printRuleInfos(
      PrintWriter printWriter, MarkdownRenderer renderer, List<RuleInfo> ruleInfos)
      throws IOException {
    for (RuleInfo ruleProto : ruleInfos) {
      printWriter.println(renderer.render(ruleProto.getRuleName(), ruleProto));
      printWriter.println();
    }
  }

  private static void printProviderInfos(
      PrintWriter printWriter, MarkdownRenderer renderer, List<ProviderInfo> providerInfos)
      throws IOException {
    for (ProviderInfo providerProto : providerInfos) {
      printWriter.println(renderer.render(providerProto.getProviderName(), providerProto));
      printWriter.println();
    }
  }

  private static void printStarlarkFunctions(
      PrintWriter printWriter,
      MarkdownRenderer renderer,
      List<StarlarkFunctionInfo> userDefinedFunctions)
      throws IOException {
    for (StarlarkFunctionInfo funcProto : userDefinedFunctions) {
      printWriter.println(renderer.render(funcProto));
      printWriter.println();
    }
  }

  private static void printAspectInfos(
      PrintWriter printWriter, MarkdownRenderer renderer, List<AspectInfo> aspectInfos)
      throws IOException {
    for (AspectInfo aspectProto : aspectInfos) {
      printWriter.println(renderer.render(aspectProto.getAspectName(), aspectProto));
      printWriter.println();
    }
  }
}
