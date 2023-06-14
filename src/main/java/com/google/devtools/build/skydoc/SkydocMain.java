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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.PackageContext;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.runfiles.Runfiles;
import com.google.devtools.build.runfiles.RunfilesForStardoc;
import com.google.devtools.build.skydoc.fakebuildapi.FakeApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDeepStructure;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;
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
import java.nio.file.Files;
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
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.Resolver;
import net.starlark.java.syntax.Resolver.Scope;
import net.starlark.java.syntax.StarlarkFile;
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
  private final String workspaceName;
  private final Runfiles.Preloaded runfiles;

  public SkydocMain(String workspaceName, Runfiles.Preloaded runfiles) {
    this.workspaceName = workspaceName;
    this.runfiles = runfiles;
  }

  public static void main(String[] args)
      throws IOException, InterruptedException, LabelSyntaxException, DocstringParseException {
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

    if (Strings.isNullOrEmpty(skydocOptions.targetFileLabel)
        || Strings.isNullOrEmpty(skydocOptions.outputFilePath)) {
      throw new IllegalArgumentException("Expected a target file label and an output file path.");
    }

    targetFileLabelString = skydocOptions.targetFileLabel;
    outputPath = skydocOptions.outputFilePath;

    Label targetFileLabel = Label.parseCanonical(targetFileLabelString);

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, ProviderInfo> providerInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctions = ImmutableMap.builder();
    ImmutableMap.Builder<String, AspectInfo> aspectInfoMap = ImmutableMap.builder();

    Module module = null;
    try {
      module =
          new SkydocMain(skydocOptions.workspaceName, Runfiles.preload())
              .eval(
                  semanticsOptions.toStarlarkSemantics(),
                  // The label passed on the command line is assumed to be canonical.
                  targetFileLabel,
                  ruleInfoMap,
                  providerInfoMap,
                  userDefinedFunctions,
                  aspectInfoMap);
    } catch (StarlarkEvaluationException | EvalException exception) {
      exception.printStackTrace();
      System.err.println("Stardoc documentation generation failed: " + exception.getMessage());
      System.exit(1);
    }

    ProtoRenderer renderer =
        render(
            module,
            ImmutableSet.copyOf(skydocOptions.symbolNames),
            ruleInfoMap.buildOrThrow(),
            providerInfoMap.buildOrThrow(),
            userDefinedFunctions.buildOrThrow(),
            aspectInfoMap.buildOrThrow());
    try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(outputPath))) {
      renderer.writeModuleInfo(out);
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
   * Renders a Starlark module to proto form.
   *
   * @param symbolNames symbols to render; if empty, all non-private symbols (i.e. those whose names
   *     do not start with '_') will be rendered.
   * @param ruleInfoMap a map of rule definition information for named rules. Keys are exported
   *     names of rules, and values are their {@link RuleInfo} rule descriptions. For example,
   *     'my_rule = rule(...)' has key 'my_rule'
   * @param providerInfoMap a map of provider definition information for named providers. Keys are
   *     exported names of providers, and values are their {@link ProviderInfo} descriptions. For
   *     example, 'my_provider = provider(...)' has key 'my_provider'
   * @param userDefinedFunctions a map of user-defined functions. Keys are exported names of
   *     functions, and values are the {@link StarlarkFunction} objects. For example, 'def
   *     my_function(foo):' is a function with key 'my_function'.
   * @param aspectInfoMap a map of aspect definition information for named aspects. Keys are
   *     exported names of aspects, and values are the {@link AspectInfo} asepct descriptions. For
   *     example, 'my_aspect = aspect(...)' has key 'my_aspect'
   */
  @VisibleForTesting
  public static ProtoRenderer render(
      Module module,
      ImmutableSet<String> symbolNames,
      ImmutableMap<String, RuleInfo> ruleInfoMap,
      ImmutableMap<String, ProviderInfo> providerInfoMap,
      ImmutableMap<String, StarlarkFunction> userDefinedFunctions,
      ImmutableMap<String, AspectInfo> aspectInfoMap)
      throws DocstringParseException {
    ImmutableMap<String, RuleInfo> filteredRuleInfos =
        ruleInfoMap.entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(toImmutableMap(Entry::getKey, Entry::getValue));
    ImmutableMap<String, ProviderInfo> filteredProviderInfos =
        providerInfoMap.entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(toImmutableMap(Entry::getKey, Entry::getValue));
    ImmutableMap<String, StarlarkFunction> filteredStarlarkFunctions =
        userDefinedFunctions.entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(toImmutableMap(Entry::getKey, Entry::getValue));
    ImmutableMap<String, AspectInfo> filteredAspectInfos =
        aspectInfoMap.entrySet().stream()
            .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
            .collect(toImmutableMap(Entry::getKey, Entry::getValue));

    String moduleDocstring = module.getDocumentation();
    if (moduleDocstring == null) {
      moduleDocstring = "";
    }
    return new ProtoRenderer()
        .appendRuleInfos(filteredRuleInfos.values())
        .appendProviderInfos(filteredProviderInfos.values())
        .appendStarlarkFunctionInfos(filteredStarlarkFunctions)
        .appendAspectInfos(filteredAspectInfos.values())
        .setModuleDocstring(moduleDocstring);
  }

  /**
   * Evaluates/interprets the Starlark file at a given path and its transitive Starlark dependencies
   * using a fake build API and collects information about all rule definitions made in the root
   * Starlark file.
   *
   * @param canonicalLabel the canonical label of the Starlark file to evaluate
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
   * @throws InterruptedException if evaluation is interrupted
   */
  @VisibleForTesting
  public Module eval(
      StarlarkSemantics semantics,
      Label canonicalLabel,
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableMap.Builder<String, ProviderInfo> providerInfoMap,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap,
      ImmutableMap.Builder<String, AspectInfo> aspectInfoMap)
      throws InterruptedException,
          IOException,
          LabelSyntaxException,
          EvalException,
          StarlarkEvaluationException {

    List<RuleInfoWrapper> ruleInfoList = new ArrayList<>();

    List<ProviderInfoWrapper> providerInfoList = new ArrayList<>();

    List<AspectInfoWrapper> aspectInfoList = new ArrayList<>();

    Module module =
        recursiveEval(semantics, canonicalLabel, ruleInfoList, providerInfoList, aspectInfoList);

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
        putStructFields(
            namespaceName, namespace, ruleFunctions, ruleInfoMap, userDefinedFunctionMap);
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
      Map<StarlarkCallable, RuleInfoWrapper> ruleFunctions,
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap)
      throws EvalException {
    for (String field : namespace.getFieldNames()) {
      String qualifiedFieldName = namespaceName + "." + field;
      if (ruleFunctions.containsKey(namespace.getValue(field))) {
        ruleInfoMap.put(
            qualifiedFieldName, ruleFunctions.get(namespace.getValue(field)).getRuleInfo().build());
      } else if (namespace.getValue(field) instanceof StarlarkFunction) {
        StarlarkFunction userDefinedFunction = (StarlarkFunction) namespace.getValue(field);
        userDefinedFunctionMap.put(qualifiedFieldName, userDefinedFunction);
      } else if (namespace.getValue(field) instanceof FakeStructApi) {
        FakeStructApi innerNamespace = (FakeStructApi) namespace.getValue(field);
        putStructFields(
            qualifiedFieldName, innerNamespace, ruleFunctions, ruleInfoMap, userDefinedFunctionMap);
      }
    }
  }

  /**
   * Recursively evaluates/interprets the Starlark file at a given path and its transitive Starlark
   * dependencies using a fake build API and collects information about all rule definitions made in
   * those files.
   *
   * @param canonicalLabel the canonical label of the Starlark file to evaluate
   * @param ruleInfoList a collection of all rule definitions made so far (using rule()); this
   *     method will add to this list as it evaluates additional files
   * @throws InterruptedException if evaluation is interrupted
   */
  private Module recursiveEval(
      StarlarkSemantics semantics,
      Label canonicalLabel,
      List<RuleInfoWrapper> ruleInfoList,
      List<ProviderInfoWrapper> providerInfoList,
      List<AspectInfoWrapper> aspectInfoList)
      throws InterruptedException, IOException, LabelSyntaxException, StarlarkEvaluationException {
    Path path = pathOfCanonicalLabel(canonicalLabel);
    String sourceRepository = canonicalLabel.getRepository().getName();

    if (pending.contains(path)) {
      throw new StarlarkEvaluationException("cycle with " + path);
    } else if (loaded.containsKey(path)) {
      return loaded.get(path);
    }
    pending.add(path);

    // Create an initial environment with a fake build API. Then use Starlark's name resolution
    // step to further populate the environment with all additional symbols not in the fake build
    // API but used by the program; these become FakeDeepStructures.
    ImmutableMap.Builder<String, Object> initialEnvBuilder = ImmutableMap.builder();
    FakeApi.addPredeclared(initialEnvBuilder, ruleInfoList, providerInfoList, aspectInfoList);
    addMorePredeclared(initialEnvBuilder);

    ImmutableMap<String, Object> initialEnv = initialEnvBuilder.build();

    Map<String, Object> predeclaredSymbols = new HashMap<>();
    predeclaredSymbols.putAll(initialEnv);

    Resolver.Module predeclaredResolver =
        (name) -> {
          if (predeclaredSymbols.containsKey(name)) {
            return Scope.PREDECLARED;
          }
          if (!Starlark.UNIVERSE.containsKey(name)) {
            predeclaredSymbols.put(name, FakeDeepStructure.create(name));
            return Scope.PREDECLARED;
          }
          return Resolver.Scope.UNIVERSAL;
        };

    // parse & compile (and get doc)
    ParserInput input = ParserInput.fromLatin1(Files.readAllBytes(path), path.toString());
    Program prog;
    try {
      StarlarkFile file = StarlarkFile.parse(input, FileOptions.DEFAULT);
      prog = Program.compileFile(file, predeclaredResolver);
    } catch (SyntaxError.Exception ex) {
      Event.replayEventsOn(eventHandler, ex.errors());
      throw new StarlarkEvaluationException(ex.getMessage());
    }

    // process loads
    Map<String, Module> imports = new HashMap<>();
    for (String load : prog.getLoads()) {
      Label apparentLoad =
          Label.parseWithPackageContext(
              load,
              PackageContext.of(
                  canonicalLabel.getPackageIdentifier(), RepositoryMapping.ALWAYS_FALLBACK));
      Label canonicalLoad = toCanonicalLabel(apparentLoad, sourceRepository);
      try {
        Module loadedModule =
            recursiveEval(semantics, canonicalLoad, ruleInfoList, providerInfoList, aspectInfoList);
        imports.put(load, loadedModule);
      } catch (NoSuchFileException noSuchFileException) {
        throw new StarlarkEvaluationException(
            String.format(
                "File %s imported '%s', yet %s was not found.",
                path, load, pathOfCanonicalLabel(canonicalLoad)),
            noSuchFileException);
      }
    }

    // execute
    Module module = Module.withPredeclared(semantics, predeclaredSymbols);
    try (Mutability mu = Mutability.create("Skydoc")) {
      StarlarkThread thread = new StarlarkThread(mu, semantics);
      // We use the default print handler, which writes to stderr.
      thread.setLoader(imports::get);
      // Fake Bazel's "export" hack, by which provider symbols
      // bound to global variables take on the name of the global variable.
      thread.setPostAssignHook(
          (name, value) -> {
            if (value instanceof FakeProviderApi) {
              ((FakeProviderApi) value).setName(name);
            }
          });

      Starlark.execFileProgram(prog, module, thread);
    } catch (EvalException ex) {
      throw new StarlarkEvaluationException(ex.getMessageWithStack());
    }

    pending.remove(path);
    loaded.put(path, module);
    return module;
  }

  private Label toCanonicalLabel(Label apparentLabel, String sourceRepository) {
    String canonicalRepositoryName =
        RunfilesForStardoc.getCanonicalRepositoryName(
            runfiles.withSourceRepository(sourceRepository),
            apparentLabel.getRepository().getName());
    return Label.parseCanonicalUnchecked(
        String.format(
            "@%s//%s:%s",
            canonicalRepositoryName,
            apparentLabel.getPackageIdentifier().getPackageFragment().getPathString(),
            apparentLabel.getName()));
  }

  private Path pathOfCanonicalLabel(Label label) {
    String runfilesDirName =
        label.getRepository().isMain() ? workspaceName : label.getRepository().getName();
    String rlocationPath = runfilesDirName + "/" + label.toPathFragment();
    return Paths.get(runfiles.unmapped().rlocation(rlocationPath));
  }

  private static void addMorePredeclared(ImmutableMap.Builder<String, Object> env) {
    // Add dummy declarations that would come from packages.StarlarkGlobals#getUtilToplevels()
    // were Skydoc allowed to depend on it. See hack for select below.
    env.put("json", Json.INSTANCE);
    env.put("proto", new ProtoModule());
    env.put(
        "depset",
        new StarlarkCallable() {
          @Override
          public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
            // Accept any arguments, return empty Depset.
            return Depset.of(Object.class, NestedSetBuilder.emptySet(Order.STABLE_ORDER));
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

  @StarlarkBuiltin(name = "ProtoModule", documented = false)
  private static final class ProtoModule implements StarlarkValue {
    @StarlarkMethod(
        name = "encode_text",
        doc = ".",
        parameters = {@Param(name = "x")})
    public String encodeText(Object x) {
      return "";
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
