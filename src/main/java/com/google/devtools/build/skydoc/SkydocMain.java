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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.skylarkbuildapi.TopLevelBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidAssetsInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidBinaryDataInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidCcLinkParamsProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidDex2OatInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidFeatureFlagSetProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidIdeInfoProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidIdlProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidLibraryAarInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidLibraryResourceClassJarProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidManifestInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidPreDexJarProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidProguardInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidSdkProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.ProguardMappingProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.UsesDataBindingProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.java.GeneratedExtensionRegistryProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.javascript.JsModuleInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.proto.ProtoBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.python.PyBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.stubs.ProviderStub;
import com.google.devtools.build.lib.skylarkbuildapi.stubs.SkylarkAspectStub;
import com.google.devtools.build.lib.skylarkbuildapi.test.TestingBootstrap;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ExpressionStatement;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Resolver;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkCallable;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.build.skydoc.fakebuildapi.FakeActionsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeBuildApiGlobals;
import com.google.devtools.build.skydoc.fakebuildapi.FakeConfigApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDefaultInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeOutputGroupInfo.FakeOutputGroupInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkNativeModuleApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkRuleFunctionsApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStarlarkAttrModuleApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStarlarkCommandLineApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi.FakeStructProviderApi;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidApplicationResourceInfo.FakeAndroidApplicationResourceInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidDeviceBrokerInfo.FakeAndroidDeviceBrokerInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidInstrumentationInfo.FakeAndroidInstrumentationInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidNativeLibsInfo.FakeAndroidNativeLibsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidResourcesInfo.FakeAndroidResourcesInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidSkylarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeApkInfo.FakeApkInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.apple.FakeAppleCommon;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigGlobalLibrary;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigSkylarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcInfo;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcModule;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcToolchainConfigInfo;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeGoWrapCcHelper;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakePyCcLinkParamsProvider;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakePyWrapCcHelper;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakePyWrapCcInfo;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaCcLinkParamsProvider;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaCommon;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaInfo.FakeJavaInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaProtoCommon;
import com.google.devtools.build.skydoc.fakebuildapi.platform.FakePlatformCommon;
import com.google.devtools.build.skydoc.fakebuildapi.proto.FakeProtoCommon;
import com.google.devtools.build.skydoc.fakebuildapi.proto.FakeProtoInfo.FakeProtoInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyInfo.FakePyInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyRuntimeInfo.FakePyRuntimeInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyStarlarkTransitions;
import com.google.devtools.build.skydoc.fakebuildapi.repository.FakeRepositoryModule;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeAnalysisFailureInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeAnalysisTestResultInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeCoverageCommon;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeTestingModule;
import com.google.devtools.build.skydoc.rendering.AspectInfoWrapper;
import com.google.devtools.build.skydoc.rendering.DocstringParseException;
import com.google.devtools.build.skydoc.rendering.ProtoRenderer;
import com.google.devtools.build.skydoc.rendering.ProviderInfoWrapper;
import com.google.devtools.build.skydoc.rendering.RuleInfoWrapper;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * Main entry point for the Skydoc binary.
 *
 * <p>Skydoc generates human-readable documentation for relevant details of Starlark files by
 * running a Starlark interpreter with a fake implementation of the build API.
 *
 * <p>Currently, Skydoc generates documentation for Starlark rule definitions (discovered by
 * invocations of the build API function {@code rule()}.
 *
 * <p>Usage:
 *
 * <pre>
 *   skydoc {target_skylark_file_label} {output_file} [symbol_name]...
 * </pre>
 *
 * <p>Generates documentation for all exported symbols of the target Starlark file that are
 * specified in the list of symbol names. If no symbol names are supplied, outputs documentation for
 * all exported symbols in the target Starlark file.
 */
public class SkydocMain {

  private final EventHandler eventHandler = new SystemOutEventHandler();
  private final LinkedHashSet<Path> pending = new LinkedHashSet<>();
  private final Map<Path, StarlarkThread> loaded = new HashMap<>();
  private final SkylarkFileAccessor fileAccessor;
  private final List<String> depRoots;
  private final String workspaceName;

  public SkydocMain(SkylarkFileAccessor fileAccessor, String workspaceName, List<String> depRoots) {
    this.fileAccessor = fileAccessor;
    this.workspaceName = workspaceName;
    if (depRoots.isEmpty()) {
      // For backwards compatibility, if no dep_roots are specified, use the current
      // directory as the only root.
      this.depRoots = ImmutableList.of(".");
    } else {
      this.depRoots = depRoots;
    }
  }

  public static void main(String[] args)
      throws IOException, InterruptedException, LabelSyntaxException, EvalException,
          DocstringParseException {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(StarlarkSemanticsOptions.class, SkydocOptions.class)
            .build();
    parser.parseAndExitUponError(args);
    StarlarkSemanticsOptions semanticsOptions = parser.getOptions(StarlarkSemanticsOptions.class);
    semanticsOptions.incompatibleDisableDeprecatedAttrParams = false;
    semanticsOptions.incompatibleNewActionsApi = false;
    SkydocOptions skydocOptions = parser.getOptions(SkydocOptions.class);

    String targetFileLabelString;
    String outputPath;
    ImmutableSet<String> symbolNames;
    ImmutableList<String> depRoots;

    if (Strings.isNullOrEmpty(skydocOptions.targetFileLabel)
        || Strings.isNullOrEmpty(skydocOptions.outputFilePath)) {
      throw new IllegalArgumentException("Expected a target file label and an output file path.");
    }

    targetFileLabelString = skydocOptions.targetFileLabel;
    outputPath = skydocOptions.outputFilePath;
    symbolNames = ImmutableSet.copyOf(skydocOptions.symbolNames);
    depRoots = ImmutableList.copyOf(skydocOptions.depRoots);

    Label targetFileLabel = Label.parseAbsolute(targetFileLabelString, ImmutableMap.of());

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, ProviderInfo> providerInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctions = ImmutableMap.builder();
    ImmutableMap.Builder<String, AspectInfo> aspectInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<Label, String> moduleDocMap = ImmutableMap.builder();

    try {
      new SkydocMain(new FilesystemFileAccessor(), skydocOptions.workspaceName, depRoots)
          .eval(
              semanticsOptions.toSkylarkSemantics(),
              targetFileLabel,
              ruleInfoMap,
              providerInfoMap,
              userDefinedFunctions,
              aspectInfoMap,
              moduleDocMap);
    } catch (StarlarkEvaluationException exception) {
      exception.printStackTrace();
      System.err.println("Stardoc documentation generation failed: " + exception.getMessage());
      System.exit(1);
    }

    Map<String, RuleInfo> filteredRuleInfos =
        ruleInfoMap.build().entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
    Map<String, ProviderInfo> filteredProviderInfos =
        providerInfoMap.build().entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
    Map<String, StarlarkFunction> filteredStarlarkFunctions =
        userDefinedFunctions.build().entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
    Map<String, AspectInfo> filteredAspectInfos =
        aspectInfoMap.build().entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));

      try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(outputPath))) {
      new ProtoRenderer()
          .appendRuleInfos(filteredRuleInfos.values())
          .appendProviderInfos(filteredProviderInfos.values())
          .appendStarlarkFunctionInfos(filteredStarlarkFunctions)
          .appendAspectInfos(filteredAspectInfos.values())
          .setModuleDocstring(moduleDocMap.build().get(targetFileLabel))
          .writeModuleInfo(out);
      }
  }

  private static boolean validSymbolName(ImmutableSet<String> symbolNames, String symbolName) {
    if (symbolNames.isEmpty()) {
      // Symbols prefixed with an underscore are private, and thus, by default, documentation
      // should not be generated for them.
      return !symbolName.startsWith("_");
    } else if (symbolNames.contains(symbolName)) {
      return true;
    } else if (symbolName.contains(".")) {
      return symbolNames.contains(symbolName.substring(0, symbolName.indexOf('.')));
    }
    return false;
  }

  /**
   * Evaluates/interprets the Starlark file at a given path and its transitive Starlark dependencies
   * using a fake build API and collects information about all rule definitions made in the root
   * Starlark file.
   *
   * @param label the label of the Starlark file to evaluate
   * @param ruleInfoMap a map builder to be populated with rule definition information for named
   *     rules. Keys are exported names of rules, and values are their {@link RuleInfo} rule
   *     descriptions. For example, 'my_rule = rule(...)' has key 'my_rule'
   * @param providerInfoMap a map builder to be populated with provider definition information for
   *     named providers. Keys are exported names of providers, and values are their {@link
   *     ProviderInfo} descriptions. For example, 'my_provider = provider(...)' has key
   *     'my_provider'
   * @param userDefinedFunctionMap a map builder to be populated with user-defined functions. Keys
   *     are exported names of functions, and values are the {@link StarlarkFunction} objects. For
   *     example, 'def my_function(foo):' is a function with key 'my_function'.
   * @param aspectInfoMap a map builder to be populated with aspect definition information for named
   *     aspects. Keys are exported names of aspects, and values are the {@link AspectInfo} asepct
   *     descriptions. For example, 'my_aspect = aspect(...)' has key 'my_aspect'
   * @param moduleDocMap a map builder to be populated with module docstrings for Starlark file.
   *     Keys are labels of Starlark files and values are their module docstrings. If the module has
   *     no docstring, an empty string will be printed.
   * @throws InterruptedException if evaluation is interrupted
   */
  public StarlarkThread eval(
      StarlarkSemantics semantics,
      Label label,
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableMap.Builder<String, ProviderInfo> providerInfoMap,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap,
      ImmutableMap.Builder<String, AspectInfo> aspectInfoMap,
      ImmutableMap.Builder<Label, String> moduleDocMap)
      throws InterruptedException, IOException, LabelSyntaxException, EvalException,
          StarlarkEvaluationException {

    List<RuleInfoWrapper> ruleInfoList = new ArrayList<>();

    List<ProviderInfoWrapper> providerInfoList = new ArrayList<>();

    List<AspectInfoWrapper> aspectInfoList = new ArrayList<>();

    StarlarkThread thread =
        recursiveEval(
            semantics, label, ruleInfoList, providerInfoList, aspectInfoList, moduleDocMap);

    Map<StarlarkCallable, RuleInfoWrapper> ruleFunctions =
        ruleInfoList.stream()
            .collect(
                Collectors.toMap(RuleInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<StarlarkCallable, ProviderInfoWrapper> providerInfos =
        providerInfoList.stream()
            .collect(Collectors.toMap(ProviderInfoWrapper::getIdentifier, Functions.identity()));

    Map<StarlarkCallable, AspectInfoWrapper> aspectFunctions =
        aspectInfoList.stream()
            .collect(
                Collectors.toMap(AspectInfoWrapper::getIdentifierFunction, Functions.identity()));

    // Sort the bindings so their ordering is deterministic.
    TreeMap<String, Object> sortedBindings =
        new TreeMap<>(thread.getGlobals().getExportedBindings());

    for (Entry<String, Object> envEntry : sortedBindings.entrySet()) {
      if (ruleFunctions.containsKey(envEntry.getValue())) {
        RuleInfo.Builder ruleInfoBuild = ruleFunctions.get(envEntry.getValue()).getRuleInfo();
        RuleInfo ruleInfo = ruleInfoBuild.setRuleName(envEntry.getKey()).build();
        ruleInfoMap.put(envEntry.getKey(), ruleInfo);
      }
      if (providerInfos.containsKey(envEntry.getValue())) {
        ProviderInfo.Builder providerInfoBuild =
            providerInfos.get(envEntry.getValue()).getProviderInfo();
        ProviderInfo providerInfo = providerInfoBuild.setProviderName(envEntry.getKey()).build();
        providerInfoMap.put(envEntry.getKey(), providerInfo);
      }
      if (envEntry.getValue() instanceof StarlarkFunction) {
        StarlarkFunction userDefinedFunction = (StarlarkFunction) envEntry.getValue();
        userDefinedFunctionMap.put(envEntry.getKey(), userDefinedFunction);
      }
      if (envEntry.getValue() instanceof FakeStructApi) {
        String namespaceName = envEntry.getKey();
        FakeStructApi namespace = (FakeStructApi) envEntry.getValue();
        putStructFields(namespaceName, namespace, userDefinedFunctionMap);
      }
      if (aspectFunctions.containsKey(envEntry.getValue())) {
        AspectInfo.Builder aspectInfoBuild =
            aspectFunctions.get(envEntry.getValue()).getAspectInfo();
        AspectInfo aspectInfo = aspectInfoBuild.setAspectName(envEntry.getKey()).build();
        aspectInfoMap.put(envEntry.getKey(), aspectInfo);
      }
    }

    return thread;
  }

  /**
   * Recursively adds functions defined in {@code namespace}, and in its nested namespaces, to
   * {@code userDefinedFunctionMap}.
   *
   * <p>Each entry's key is the fully qualified function name, e.g. {@code
   * "outernamespace.innernamespace.func"}. {@code namespaceName} is the fully qualified name of
   * {@code namespace} itself.
   */
  private static void putStructFields(
      String namespaceName,
      FakeStructApi namespace,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap)
      throws EvalException {
    for (String field : namespace.getFieldNames()) {
      String qualifiedFieldName = namespaceName + "." + field;
      if (namespace.getValue(field) instanceof StarlarkFunction) {
        StarlarkFunction userDefinedFunction = (StarlarkFunction) namespace.getValue(field);
        userDefinedFunctionMap.put(qualifiedFieldName, userDefinedFunction);
      } else if (namespace.getValue(field) instanceof FakeStructApi) {
        FakeStructApi innerNamespace = (FakeStructApi) namespace.getValue(field);
        putStructFields(qualifiedFieldName, innerNamespace, userDefinedFunctionMap);
      }
    }
  }

  private static String getModuleDoc(StarlarkFile buildFileAST) {
    ImmutableList<Statement> fileStatements = buildFileAST.getStatements();
    if (!fileStatements.isEmpty()) {
      Statement stmt = fileStatements.get(0);
      if (stmt instanceof ExpressionStatement) {
        Expression expr = ((ExpressionStatement) stmt).getExpression();
        if (expr instanceof StringLiteral) {
          return ((StringLiteral) expr).getValue();
        }
      }
    }
    return "";
  }

  /**
   * Recursively evaluates/interprets the Starlark file at a given path and its transitive skylark
   * dependencies using a fake build API and collects information about all rule definitions made in
   * those files.
   *
   * @param label the label of the Starlark file to evaluate
   * @param ruleInfoList a collection of all rule definitions made so far (using rule()); this
   *     method will add to this list as it evaluates additional files
   * @throws InterruptedException if evaluation is interrupted
   */
  private StarlarkThread recursiveEval(
      StarlarkSemantics semantics,
      Label label,
      List<RuleInfoWrapper> ruleInfoList,
      List<ProviderInfoWrapper> providerInfoList,
      List<AspectInfoWrapper> aspectInfoList,
      ImmutableMap.Builder<Label, String> moduleDocMap)
      throws InterruptedException, IOException, LabelSyntaxException, StarlarkEvaluationException {
    Path path = pathOfLabel(label, semantics);

    if (pending.contains(path)) {
      throw new StarlarkEvaluationException("cycle with " + path);
    } else if (loaded.containsKey(path)) {
      return loaded.get(path);
    }
    pending.add(path);

    ParserInput parserInputSource = getInputSource(path.toString());
    StarlarkFile file = StarlarkFile.parse(parserInputSource);
    Event.replayEventsOn(eventHandler, file.errors());

    moduleDocMap.put(label, getModuleDoc(file));

    Map<String, Module> imports = new HashMap<>();
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        LoadStatement load = (LoadStatement) stmt;
        String module = load.getImport().getValue();
        Label relativeLabel = label.getRelativeWithRemapping(module, ImmutableMap.of());
        try {
          StarlarkThread importThread =
              recursiveEval(
                  semantics,
                  relativeLabel,
                  ruleInfoList,
                  providerInfoList,
                  aspectInfoList,
                  moduleDocMap);
          imports.put(module, importThread.getGlobals());
        } catch (NoSuchFileException noSuchFileException) {
          throw new StarlarkEvaluationException(
              String.format(
                  "File %s imported '%s', yet %s was not found, even at roots %s.",
                  path, module, pathOfLabel(relativeLabel, semantics), depRoots),
              noSuchFileException);
        }
      }
    }

    StarlarkThread thread =
        evalSkylarkBody(semantics, file, imports, ruleInfoList, providerInfoList, aspectInfoList);

    pending.remove(path);
    thread.mutability().freeze();
    loaded.put(path, thread);
    return thread;
  }

  private Path pathOfLabel(Label label, StarlarkSemantics semantics) {
    String workspacePrefix = "";
    if (!label.getWorkspaceRoot(semantics).isEmpty()
        && !label.getWorkspaceName().equals(workspaceName)) {
      workspacePrefix = label.getWorkspaceRoot(semantics) + "/";
    }

    return Paths.get(workspacePrefix + label.toPathFragment());
  }

  private ParserInput getInputSource(String bzlWorkspacePath) throws IOException {
    for (String rootPath : depRoots) {
      if (fileAccessor.fileExists(rootPath + "/" + bzlWorkspacePath)) {
        return fileAccessor.inputSource(rootPath + "/" + bzlWorkspacePath);
      }
    }

    // All depRoots attempted and no valid file was found.
    throw new NoSuchFileException(bzlWorkspacePath);
  }

  /** Evaluates the AST from a single Starlark file, given the already-resolved imports. */
  private StarlarkThread evalSkylarkBody(
      StarlarkSemantics semantics,
      StarlarkFile file,
      Map<String, Module> imports,
      List<RuleInfoWrapper> ruleInfoList,
      List<ProviderInfoWrapper> providerInfoList,
      List<AspectInfoWrapper> aspectInfoList)
      throws InterruptedException, StarlarkEvaluationException {

    StarlarkThread thread =
        createStarlarkThread(
            semantics,
            globalFrame(ruleInfoList, providerInfoList, aspectInfoList),
            imports);
    Module module = thread.getGlobals();

    Resolver.resolveFile(file, module);
    if (!file.ok()) {
      throw new StarlarkEvaluationException(file.errors().get(0).toString());
    }

    try {
      EvalUtils.exec(file, module, thread);
    } catch (EvalException | InterruptedException ex) {
      // This exception class seems a bit unnecessary. Replace with EvalException?
      throw new StarlarkEvaluationException("Starlark evaluation error", ex);
    }

    thread.mutability().freeze();

    return thread;
  }

  /**
   * Initialize and return a global frame containing the fake build API.
   *
   * @param ruleInfoList the list of {@link RuleInfo} objects, to which rule() invocation
   *     information will be added
   * @param providerInfoList the list of {@link ProviderInfo} objects, to which provider()
   *     invocation information will be added
   */
  private static Module globalFrame(
      List<RuleInfoWrapper> ruleInfoList,
      List<ProviderInfoWrapper> providerInfoList,
      List<AspectInfoWrapper> aspectInfoList) {
    TopLevelBootstrap topLevelBootstrap =
        new TopLevelBootstrap(
            new FakeBuildApiGlobals(),
            new FakeStarlarkAttrModuleApi(),
            new FakeStarlarkCommandLineApi(),
            new FakeSkylarkNativeModuleApi(),
            new FakeSkylarkRuleFunctionsApi(ruleInfoList, providerInfoList, aspectInfoList),
            new FakeStructProviderApi(),
            new FakeOutputGroupInfoProvider(),
            new FakeActionsInfoProvider(),
            new FakeDefaultInfoProvider());
    AndroidBootstrap androidBootstrap =
        new AndroidBootstrap(
            new FakeAndroidSkylarkCommon(),
            new FakeApkInfoProvider(),
            new FakeAndroidInstrumentationInfoProvider(),
            new FakeAndroidDeviceBrokerInfoProvider(),
            new FakeAndroidResourcesInfoProvider(),
            new FakeAndroidNativeLibsInfoProvider(),
            new FakeAndroidApplicationResourceInfoProvider());
    AppleBootstrap appleBootstrap = new AppleBootstrap(new FakeAppleCommon());
    ConfigBootstrap configBootstrap =
        new ConfigBootstrap(
            new FakeConfigSkylarkCommon(), new FakeConfigApi(), new FakeConfigGlobalLibrary());
    CcBootstrap ccBootstrap =
        new CcBootstrap(
            new FakeCcModule(),
            new FakeCcInfo.Provider(),
            new FakeCcToolchainConfigInfo.Provider(),
            new FakePyWrapCcHelper(),
            new FakeGoWrapCcHelper(),
            new FakePyWrapCcInfo.Provider(),
            new FakePyCcLinkParamsProvider.Provider());
    JavaBootstrap javaBootstrap =
        new JavaBootstrap(
            new FakeJavaCommon(),
            new FakeJavaInfoProvider(),
            new FakeJavaProtoCommon(),
            new FakeJavaCcLinkParamsProvider.Provider());
    PlatformBootstrap platformBootstrap = new PlatformBootstrap(new FakePlatformCommon());
    ProtoBootstrap protoBootstrap =
        new ProtoBootstrap(
            new FakeProtoInfoProvider(),
            new FakeProtoCommon(),
            new SkylarkAspectStub(),
            new ProviderStub());
    PyBootstrap pyBootstrap =
        new PyBootstrap(
            new FakePyInfoProvider(),
            new FakePyRuntimeInfoProvider(),
            new FakePyStarlarkTransitions());
    RepositoryBootstrap repositoryBootstrap =
        new RepositoryBootstrap(new FakeRepositoryModule(ruleInfoList));
    TestingBootstrap testingBootstrap =
        new TestingBootstrap(
            new FakeTestingModule(),
            new FakeCoverageCommon(),
            new FakeAnalysisFailureInfoProvider(),
            new FakeAnalysisTestResultInfoProvider());

    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();

    envBuilder.putAll(Starlark.UNIVERSE);

    // Add stub declarations for Blaze-only things as a quick fix
    // for a broken test; see b/155126966 and b/155178103.
    // TODO(adonovan): fix properly ASAP.
    for (String name :
        new String[] {
          "DataBindingV2Info",
          "PintoModuleLegacyDepsMgmtProvider",
          "ProguardSpecProvider",
          "js_common",
          "pkg_common",
        }) {
      envBuilder.put(name, Starlark.NONE);
    }

    // Add dummy declarations that would come from packages.StarlarkLibrary.COMMON
    // were Skydoc allowed to depend on it. See hack for select below.
    envBuilder.put(
        "depset",
        new StarlarkCallable() {
          @Override
          public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
            // Accept any arguments, return empty Depset.
            return Depset.of(
                Depset.ElementType.EMPTY, NestedSetBuilder.emptySet(Order.STABLE_ORDER));
          }

          @Override
          public String getName() {
            return "depset";
          }
        });

    // Declare a fake implementation of select that just returns the first
    // value in the dict. (This program is forbidden from depending on the real
    // implementation of 'select' in lib.packages, and so the hacks multiply.)
    envBuilder.put(
        "select",
        new StarlarkCallable() {
          @Override
          public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
              throws EvalException {
            for (Map.Entry<?, ?> e : ((Dict<?, ?>) positional[0]).entrySet()) {
              return e.getValue();
            }
            throw Starlark.errorf("select: empty dict");
          }

          @Override
          public String getName() {
            return "select";
          }
        });

    topLevelBootstrap.addBindingsToBuilder(envBuilder);
    androidBootstrap.addBindingsToBuilder(envBuilder);
    appleBootstrap.addBindingsToBuilder(envBuilder);
    ccBootstrap.addBindingsToBuilder(envBuilder);
    configBootstrap.addBindingsToBuilder(envBuilder);
    javaBootstrap.addBindingsToBuilder(envBuilder);
    platformBootstrap.addBindingsToBuilder(envBuilder);
    protoBootstrap.addBindingsToBuilder(envBuilder);
    pyBootstrap.addBindingsToBuilder(envBuilder);
    repositoryBootstrap.addBindingsToBuilder(envBuilder);
    testingBootstrap.addBindingsToBuilder(envBuilder);
    addNonBootstrapGlobals(envBuilder);

    return Module.createForBuiltins(envBuilder.build());
  }

  // TODO(cparsons): Remove this constant by migrating the contained symbols to bootstraps.
  private static final String[] nonBootstrapGlobals = {
    "android_data",
    AndroidDex2OatInfoApi.NAME,
    AndroidManifestInfoApi.NAME,
    AndroidAssetsInfoApi.NAME,
    AndroidLibraryAarInfoApi.NAME,
    AndroidProguardInfoApi.NAME,
    AndroidIdlProviderApi.NAME,
    AndroidIdeInfoProviderApi.NAME,
    AndroidPreDexJarProviderApi.NAME,
    UsesDataBindingProviderApi.NAME,
    AndroidCcLinkParamsProviderApi.NAME,
    AndroidLibraryResourceClassJarProviderApi.NAME,
    AndroidSdkProviderApi.NAME,
    AndroidFeatureFlagSetProviderApi.NAME,
    ProguardMappingProviderApi.NAME,
    GeneratedExtensionRegistryProviderApi.NAME,
    AndroidBinaryDataInfoApi.NAME,
    JsModuleInfoApi.NAME,
    "JsInfo",
    "PintoModuleProvider"
  };

  /**
   * A hack to add a number of global symbols which are part of the build API but are otherwise
   * added by Bazel.
   */
  // TODO(cparsons): Remove this method by migrating the contained symbols to bootstraps.
  private static void addNonBootstrapGlobals(ImmutableMap.Builder<String, Object> envBuilder) {
    for (String global : nonBootstrapGlobals) {
      envBuilder.put(global, global);
    }
  }

  private static StarlarkThread createStarlarkThread(
      StarlarkSemantics semantics, Module globals, Map<String, Module> imports) {
    // We use the default print handler, which writes to stderr.
    StarlarkThread thread =
        StarlarkThread.builder(Mutability.create("Skydoc"))
            .setSemantics(semantics)
            .setGlobals(globals)
            .build();
    thread.setLoader(imports::get);
    return thread;
  }

  /** Exception thrown when Starlark evaluation fails (due to malformed Starlark). */
  @VisibleForTesting
  static class StarlarkEvaluationException extends Exception {
    public StarlarkEvaluationException(String message) {
      super(message);
    }

    public StarlarkEvaluationException(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
