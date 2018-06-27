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

package com.google.devtools.build.skydoc;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skylarkbuildapi.TopLevelBootstrap;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skydoc.fakebuildapi.FakeActionsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeBuildApiGlobals;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDefaultInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeOutputGroupInfo.FakeOutputGroupInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkAttrApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkCommandLineApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkNativeModuleApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkRuleFunctionsApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi.FakeStructProviderApi;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

/**
 * Main entry point for the Skydoc binary.
 *
 * <p>Skydoc generates human-readable documentation for relevant details of skylark files by
 * running a skylark interpreter with a fake implementation of the build API.</p>
 *
 * <p>Currently, Skydoc generates documentation for skylark rule definitions (discovered by
 * invocations of the build API function {@code rule()}.</p>
 *
 * <p>Usage:</p>
 * <pre>
 *   skydoc {target_skylark_file} {output_file}
 * </pre>
 */
public class SkydocMain {

  private final EventHandler eventHandler = new SystemOutEventHandler();

  public static void main(String[] args) throws IOException, InterruptedException {
    if (args.length != 2) {
      throw new IllegalArgumentException("Expected two arguments. Usage:\n"
          + "{skydoc_bin} {target_skylark_file} {output_file}");
    }

    String bzlPath = args[0];
    String outputPath = args[1];

    Path path = Paths.get(bzlPath);
    byte[] content = Files.readAllBytes(path);

    ParserInputSource parserInputSource =
        ParserInputSource.create(content, PathFragment.create(path.toString()));

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
    ImmutableList.Builder<RuleInfo> unexportedRuleInfos = ImmutableList.builder();

    new SkydocMain().eval(parserInputSource, ruleInfoMap, unexportedRuleInfos);

    try (PrintWriter printWriter = new PrintWriter(outputPath, "UTF-8")) {
      printRuleInfos(printWriter, ruleInfoMap.build(), unexportedRuleInfos.build());
    }
  }

  // TODO(cparsons): Improve output (markdown or HTML).
  private static void printRuleInfos(
      PrintWriter printWriter,
      Map<String, RuleInfo> ruleInfos,
      List<RuleInfo> unexportedRuleInfos) throws IOException {
    for (Entry<String, RuleInfo> ruleInfoEntry : ruleInfos.entrySet()) {
      printRuleInfo(printWriter, ruleInfoEntry.getKey(), ruleInfoEntry.getValue());
    }
    for (RuleInfo unexportedRuleInfo : unexportedRuleInfos) {
      printRuleInfo(printWriter, "<unknown name>", unexportedRuleInfo);
    }
  }

  private static void printRuleInfo(
      PrintWriter printWriter, String exportedName, RuleInfo ruleInfo) {
    printWriter.println(exportedName);
    printWriter.println(ruleInfo.getDescription());
    printWriter.println();
  }

  /**
   * Evaluates/interprets the skylark file at the given input source using a fake build API and
   * collects information about all rule definitions made in that file.
   *
   * @param parserInputSource the input source representing the input skylark file
   * @param ruleInfoMap a map builder to be populated with rule definition information for
   *     named rules. Keys are exported names of rules, and values are their {@link RuleInfo}
   *     rule descriptions. For example, 'my_rule = rule(...)' has key 'my_rule'
   * @param unexportedRuleInfos a list builder to be populated with rule definition information
   *     for rules which were not exported as top level symbols
   * @throws InterruptedException if evaluation is interrupted
   */
  // TODO(cparsons): Evaluate load statements recursively.
  public void eval(ParserInputSource parserInputSource,
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableList.Builder<RuleInfo> unexportedRuleInfos)
      throws InterruptedException {
    List<RuleInfo> ruleInfoList = new ArrayList<>();

    BuildFileAST buildFileAST = BuildFileAST.parseSkylarkFile(
        parserInputSource, eventHandler);

    Environment env = createEnvironment(
        eventHandler,
        globalFrame(ruleInfoList),
        /* imports= */ ImmutableMap.of());

    if (!buildFileAST.exec(env, eventHandler)) {
      throw new RuntimeException("Error loading file");
    }

    env.mutability().freeze();

    Map<BaseFunction, RuleInfo> ruleFunctions = ruleInfoList.stream()
        .collect(Collectors.toMap(
            RuleInfo::getIdentifierFunction,
            Functions.identity()));

    for (Entry<String, Object> envEntry : env.getGlobals().getBindings().entrySet()) {
      if (ruleFunctions.containsKey(envEntry.getValue())) {
        ruleInfoMap.put(envEntry.getKey(), ruleFunctions.get(envEntry.getValue()));
        ruleFunctions.remove(envEntry.getValue());
      }
    }

    unexportedRuleInfos.addAll(ruleFunctions.values());
  }

  /**
   * Initialize and return a global frame containing the fake build API.
   *
   * @param ruleInfoList the list of {@link RuleInfo} objects, to which rule() invocation
   *     information will be added
   */
  private static GlobalFrame globalFrame(List<RuleInfo> ruleInfoList) {
    // TODO(cparsons): Complete the Fake Build API stubs. For example, implement provider(),
    // and include the other bootstraps.
    TopLevelBootstrap topLevelBootstrap =
        new TopLevelBootstrap(new FakeBuildApiGlobals(),
            new FakeSkylarkAttrApi(),
            new FakeSkylarkCommandLineApi(),
            new FakeSkylarkNativeModuleApi(),
            new FakeSkylarkRuleFunctionsApi(ruleInfoList),
            new FakeStructProviderApi(),
            new FakeOutputGroupInfoProvider(),
            new FakeActionsInfoProvider(),
            new FakeDefaultInfoProvider());

    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();

    Runtime.addConstantsToBuilder(envBuilder);
    MethodLibrary.addBindingsToBuilder(envBuilder);
    topLevelBootstrap.addBindingsToBuilder(envBuilder);

    return GlobalFrame.createForBuiltins(envBuilder.build());
  }

  private static Environment createEnvironment(EventHandler eventHandler, GlobalFrame globals,
      Map<String, Extension> imports) {
    return Environment.builder(Mutability.create("Skydoc"))
        .useDefaultSemantics()
        .setGlobals(globals)
        .setImportedExtensions(imports)
        .setEventHandler(eventHandler)
        .build();
  }
}
