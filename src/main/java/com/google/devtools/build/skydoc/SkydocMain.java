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
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidDex2OatInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.UsesDataBindingProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.GeneratedExtensionRegistryProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaNativeLibraryInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.javascript.JsModuleInfoApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi;
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
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.lib.json.Json;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ExpressionStatement;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.Statement;
import net.starlark.java.syntax.StringLiteral;
import net.starlark.java.syntax.SyntaxError;

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
 *   skydoc {target_starlark_file_label} {output_file} [symbol_name]...
 * </pre>
 *
 * <p>Generates documentation for all exported symbols of the target Starlark file that are
 * specified in the list of symbol names. If no symbol names are supplied, outputs documentation for
 * all exported symbols in the target Starlark file.
 */
public class SkydocMain {

  private final EventHandler eventHandler = new SystemOutEventHandler();
  private final LinkedHashSet<Path> pending = new LinkedHashSet<>();
  private final Map<Path, Module> loaded = new HashMap<>();
  private final StarlarkFileAccessor fileAccessor;
  private final List<String> depRoots;
  private final String workspaceName;

  public SkydocMain(
      StarlarkFileAccessor fileAccessor, String workspaceName, List<String> depRoots) {
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
            .optionsClasses(BuildLanguageOptions.class, SkydocOptions.class)
            .build();
    parser.parseAndExitUponError(args);
    BuildLanguageOptions semanticsOptions = parser.getOptions(BuildLanguageOptions.class);
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
              semanticsOptions.toStarlarkSemantics(),
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
  public Module eval(
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

    Module module =
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

    // Sort the globals bindings by name.
    TreeMap<String, Object> sortedBindings = new TreeMap<>(module.getGlobals());

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

    return module;
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
   * Recursively evaluates/interprets the Starlark file at a given path and its transitive Starlark
   * dependencies using a fake build API and collects information about all rule definitions made in
   * those files.
   *
   * @param label the label of the Starlark file to evaluate
   * @param ruleInfoList a collection of all rule definitions made so far (using rule()); this
   *     method will add to this list as it evaluates additional files
   * @throws InterruptedException if evaluation is interrupted
   */
  private Module recursiveEval(
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

    // Add fake build API.
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    FakeApi.addPredeclared(env, ruleInfoList, providerInfoList, aspectInfoList);
    addMorePredeclared(env);
    Module module = Module.withPredeclared(semantics, env.build());

    // parse & compile (and get doc)
    ParserInput input = getInputSource(path.toString());
    Program prog;
    try {
      StarlarkFile file = StarlarkFile.parse(input, FileOptions.DEFAULT);
      moduleDocMap.put(label, getModuleDoc(file));
      prog = Program.compileFile(file, module);
    } catch (SyntaxError.Exception ex) {
      Event.replayEventsOn(eventHandler, ex.errors());
      throw new StarlarkEvaluationException(ex.getMessage());
    }

    // process loads
    Map<String, Module> imports = new HashMap<>();
    for (String load : prog.getLoads()) {
      Label relativeLabel = label.getRelativeWithRemapping(load, ImmutableMap.of());
      try {
        Module loadedModule =
            recursiveEval(
                semantics,
                relativeLabel,
                ruleInfoList,
                providerInfoList,
                aspectInfoList,
                moduleDocMap);
        imports.put(load, loadedModule);
      } catch (NoSuchFileException noSuchFileException) {
        throw new StarlarkEvaluationException(
            String.format(
                "File %s imported '%s', yet %s was not found, even at roots %s.",
                path, load, pathOfLabel(relativeLabel, semantics), depRoots),
            noSuchFileException);
      }
    }

    // execute
    try (Mutability mu = Mutability.create("Skydoc")) {
      StarlarkThread thread = new StarlarkThread(mu, semantics);
      // We use the default print handler, which writes to stderr.
      thread.setLoader(imports::get);

      Starlark.execFileProgram(prog, module, thread);
    } catch (EvalException ex) {
      throw new StarlarkEvaluationException(ex.getMessageWithStack());
    }

    pending.remove(path);
    loaded.put(path, module);
    return module;
  }

  private Path pathOfLabel(Label label, StarlarkSemantics semantics) {
    String workspacePrefix = "";
    if (!label.getWorkspaceRootForStarlarkOnly(semantics).isEmpty()
        && !label.getWorkspaceName().equals(workspaceName)) {
      workspacePrefix = label.getWorkspaceRootForStarlarkOnly(semantics) + "/";
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

  private static void addMorePredeclared(ImmutableMap.Builder<String, Object> env) {
    addNonBootstrapGlobals(env);

    // Add dummy declarations that would come from packages.StarlarkLibrary.COMMON
    // were Skydoc allowed to depend on it. See hack for select below.
    env.put("json", Json.INSTANCE);
    env.put("proto", new ProtoModule());
    env.put(
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
    env.put(
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
  }

  // TODO(cparsons): Remove this constant by migrating the contained symbols to bootstraps.
  private static final String[] nonBootstrapGlobals = {
    "android_data",
    AndroidDex2OatInfoApi.NAME,
    UsesDataBindingProviderApi.NAME,
    GeneratedExtensionRegistryProviderApi.NAME,
    JavaNativeLibraryInfoApi.NAME,
    JsModuleInfoApi.NAME,
    "JsInfo",
    "js_common",
    "pkg_common",
  };

  @StarlarkBuiltin(name = "ProtoModule", doc = "")
  private static final class ProtoModule implements StarlarkValue {
    @StarlarkMethod(
        name = "encode_text",
        doc = ".",
        parameters = {@Param(name = "x")})
    public String encodeText(Object x) {
      return "";
    }
  }

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
