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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.TopLevelBootstrap;
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

  // Pattern to match the assignment of a variable to a rule definition
  // For example, 'my_rule = rule(' will match and have 'my_rule' available as group(1).
  private static final Pattern ruleDefinitionLinePattern =
      Pattern.compile("([^\\s]+) = rule\\(");

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

    List<RuleInfo> ruleInfoList = new SkydocMain().eval(parserInputSource);

    try (PrintWriter printWriter = new PrintWriter(outputPath, "UTF-8")) {
      printRuleInfos(printWriter, ruleInfoList);
    }
  }

  // TODO(cparsons): Improve output (markdown or HTML).
  private static void printRuleInfos(
      PrintWriter printWriter, List<RuleInfo> ruleInfos) throws IOException {
    for (RuleInfo ruleInfo : ruleInfos) {
      Location location = ruleInfo.getLocation();
      Path filePath = Paths.get(location.getPath().getPathString());
      List<String> lines = Files.readAllLines(filePath, UTF_8);
      String definingString = lines.get(location.getStartLine() - 1);
      // Rule definitions don't specify their own visible name directly. Instead, the name of
      // a rule is dependent on the name of the variable assigend to the return value of rule().
      // This attempts to find a line of the form 'foo = rule(' and thus label the rule as
      // named 'foo'.
      // TODO(cparsons): Inspect the global bindings of the environment instead of using string
      // matching.
      Matcher matcher = ruleDefinitionLinePattern.matcher(definingString);
      if (matcher.matches()) {
        printWriter.println(matcher.group(1));
      } else {
        printWriter.println("<unknown name>");
      }
      printWriter.println(ruleInfo.getDescription());
    }
  }

  /**
   * Evaluates/interprets the skylark file at the given input source using a fake build API and
   * collects information about all rule definitions made in that file.
   *
   * @param parserInputSource the input source representing the input skylark file
   * @return a list of {@link RuleInfo} objects describing the rule definitions
   * @throws InterruptedException if evaluation is interrupted
   */
  // TODO(cparsons): Evaluate load statements recursively.
  public List<RuleInfo> eval(ParserInputSource parserInputSource)
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

    return ruleInfoList;
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
