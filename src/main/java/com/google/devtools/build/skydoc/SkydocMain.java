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
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.test.TestingBootstrap;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkImport;
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
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidDeviceBrokerInfo.FakeAndroidDeviceBrokerInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidInstrumentationInfo.FakeAndroidInstrumentationInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidNativeLibsInfo.FakeAndroidNativeLibsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidResourcesInfo.FakeAndroidResourcesInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidSkylarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeApkInfo.FakeApkInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.apple.FakeAppleCommon;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigSkylarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcModule;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaCommon;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaInfo.FakeJavaInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaProtoCommon;
import com.google.devtools.build.skydoc.fakebuildapi.platform.FakePlatformCommon;
import com.google.devtools.build.skydoc.fakebuildapi.repository.FakeRepositoryModule;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeTestingModule;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
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
  private final LinkedHashSet<Path> pending = new LinkedHashSet<>();
  private final Map<Path, Environment> loaded = new HashMap<>();
  private final SkylarkFileAccessor fileAccessor;

  public SkydocMain(SkylarkFileAccessor fileAccessor) {
    this.fileAccessor = fileAccessor;
  }

  public static void main(String[] args) throws IOException, InterruptedException {
    if (args.length != 2) {
      throw new IllegalArgumentException("Expected two arguments. Usage:\n"
          + "{skydoc_bin} {target_skylark_file} {output_file}");
    }

    String bzlPath = args[0];
    String outputPath = args[1];

    Path path = Paths.get(bzlPath);

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
    ImmutableList.Builder<RuleInfo> unexportedRuleInfos = ImmutableList.builder();

    new SkydocMain(new FilesystemFileAccessor()).eval(path, ruleInfoMap, unexportedRuleInfos);

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
   * Evaluates/interprets the skylark file at a given path and its transitive skylark dependencies
   * using a fake build API and collects information about all rule definitions made in those files.
   *
   * @param path the path of the skylark file to evaluate
   * @param ruleInfoMap a map builder to be populated with rule definition information for
   *     named rules. Keys are exported names of rules, and values are their {@link RuleInfo}
   *     rule descriptions. For example, 'my_rule = rule(...)' has key 'my_rule'
   * @param unexportedRuleInfos a list builder to be populated with rule definition information
   *     for rules which were not exported as top level symbols
   * @throws InterruptedException if evaluation is interrupted
   */
  public Environment eval(
      Path path,
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableList.Builder<RuleInfo> unexportedRuleInfos)
      throws InterruptedException, IOException {
    if (pending.contains(path)) {
      throw new IllegalStateException("cycle with " + path);
    } else if (loaded.containsKey(path)) {
      return loaded.get(path);
    }
    pending.add(path);

    ParserInputSource parserInputSource = fileAccessor.inputSource(path.toString());
    BuildFileAST buildFileAST = BuildFileAST.parseSkylarkFile(parserInputSource, eventHandler);

    Map<String, Extension> imports = new HashMap<>();
    for (SkylarkImport anImport : buildFileAST.getImports()) {
      Path importPath = fromPathFragment(path, anImport.asPathFragment());

      Environment importEnv = eval(importPath, ruleInfoMap, unexportedRuleInfos);

      imports.put(anImport.getImportString(), new Extension(importEnv));
    }

    Environment env = evalSkylarkBody(buildFileAST, imports, ruleInfoMap, unexportedRuleInfos);

    pending.remove(path);
    env.mutability().freeze();
    loaded.put(path, env);
    return env;
  }

  private static Path fromPathFragment(Path fromPath, PathFragment pathFragment) {
    return pathFragment.isAbsolute()
        ? Paths.get(pathFragment.getPathString())
        : fromPath.resolveSibling(pathFragment.getPathString());
  }

  /**
   * Evaluates the AST from a single skylark file, given the already-resolved imports.
   */
  private Environment evalSkylarkBody(
      BuildFileAST buildFileAST,
      Map<String, Extension> imports,
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableList.Builder<RuleInfo> unexportedRuleInfos) throws InterruptedException {
    List<RuleInfo> ruleInfoList = new ArrayList<>();

    Environment env = createEnvironment(
        eventHandler,
        globalFrame(ruleInfoList),
        imports);

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

    return env;
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
    AndroidBootstrap androidBootstrap = new AndroidBootstrap(new FakeAndroidSkylarkCommon(),
        new FakeApkInfoProvider(),
        new FakeAndroidInstrumentationInfoProvider(),
        new FakeAndroidDeviceBrokerInfoProvider(),
        new FakeAndroidResourcesInfoProvider(),
        new FakeAndroidNativeLibsInfoProvider());
    AppleBootstrap appleBootstrap = new AppleBootstrap(new FakeAppleCommon());
    ConfigBootstrap configBootstrap = new ConfigBootstrap(new FakeConfigSkylarkCommon());
    CcBootstrap ccBootstrap = new CcBootstrap(new FakeCcModule());
    JavaBootstrap javaBootstrap = new JavaBootstrap(new FakeJavaCommon(),
        new FakeJavaInfoProvider(),
        new FakeJavaProtoCommon());
    PlatformBootstrap platformBootstrap = new PlatformBootstrap(new FakePlatformCommon());
    RepositoryBootstrap repositoryBootstrap = new RepositoryBootstrap(new FakeRepositoryModule());
    TestingBootstrap testingBootstrap = new TestingBootstrap(new FakeTestingModule());

    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();

    Runtime.addConstantsToBuilder(envBuilder);
    MethodLibrary.addBindingsToBuilder(envBuilder);
    topLevelBootstrap.addBindingsToBuilder(envBuilder);
    androidBootstrap.addBindingsToBuilder(envBuilder);
    appleBootstrap.addBindingsToBuilder(envBuilder);
    ccBootstrap.addBindingsToBuilder(envBuilder);
    configBootstrap.addBindingsToBuilder(envBuilder);
    javaBootstrap.addBindingsToBuilder(envBuilder);
    platformBootstrap.addBindingsToBuilder(envBuilder);
    repositoryBootstrap.addBindingsToBuilder(envBuilder);
    testingBootstrap.addBindingsToBuilder(envBuilder);

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
