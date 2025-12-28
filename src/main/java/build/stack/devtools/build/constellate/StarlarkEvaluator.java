package build.stack.devtools.build.constellate;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.Stack;
import java.util.Map;
import java.util.Collection;
import java.util.Map.Entry;
import java.util.concurrent.locks.ReentrantLock;
import java.util.TreeMap;
import java.util.stream.Collectors;

import build.stack.devtools.build.constellate.fakebuildapi.FakeApi;
import build.stack.devtools.build.constellate.fakebuildapi.FakeDeepStructure;
import build.stack.devtools.build.constellate.fakebuildapi.FakeProviderApi;
import build.stack.devtools.build.constellate.fakebuildapi.FakeStarlarkRuleFunctionsApi;
import build.stack.devtools.build.constellate.fakebuildapi.PostAssignHookAssignableIdentifier;
import build.stack.devtools.build.constellate.fakebuildapi.FakeStructApi;
import build.stack.devtools.build.constellate.RealObjectEnhancer;
import build.stack.devtools.build.constellate.rendering.AspectInfoWrapper;
import build.stack.devtools.build.constellate.rendering.DocstringParseException;
import build.stack.devtools.build.constellate.rendering.FunctionUtil;
import build.stack.devtools.build.constellate.rendering.MacroInfoWrapper;
import build.stack.devtools.build.constellate.rendering.ModuleExtensionInfoWrapper;
import build.stack.devtools.build.constellate.rendering.ProviderInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RepositoryRuleInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RuleInfoWrapper;

import build.stack.starlark.v1beta1.StarlarkProtos;
// import build.stack.starlark.v1beta1.StarlarkProtos.Binding;

import com.google.devtools.build.lib.starlarkdocextract.ExtractionException;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.OriginKey;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;

import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Tuple;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.lib.json.Json;
import net.starlark.java.syntax.Argument;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.CallExpression;
import net.starlark.java.syntax.DotExpression;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ExpressionStatement;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.Node;
import net.starlark.java.syntax.NodeVisitor;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.LoadStatement;
import net.starlark.java.syntax.Resolver;
import net.starlark.java.syntax.Resolver.Scope;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.Statement;
import net.starlark.java.syntax.StringLiteral;
import net.starlark.java.syntax.SyntaxError;
import net.starlark.java.types.StarlarkType;

/**
 * Main entry point for the Constellate program.
 *
 * <p>
 * Constellate generates human-readable documentation for relevant details of
 * Starlark files by running a Starlark interpreter with a fake implementation
 * of the build API.
 *
 * <p>
 * Currently, Constellate generates documentation for Starlark rule definitions
 * (discovered by invocations of the build API function {@code rule()}.
 *
 * <p>
 * Usage:
 *
 * <pre>
 *   Constellate {target_starlark_file_label} {output_file} [symbol_name]...
 * </pre>
 *
 * <p>
 * Generates documentation for all exported symbols of the target Starlark file
 * that are specified in the list of symbol names. If no symbol names are
 * supplied, outputs documentation for all exported symbols in the target
 * Starlark file.
 */
public class StarlarkEvaluator {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  // eventHandler is used to replay events when we get a compile error.
  private final EventHandler eventHandler = new SystemOutEventHandler();
  // fileAccessor helps load files.
  private final StarlarkFileAccessor fileAccessor;
  // depRoots is the list of module root dirs.
  private final List<String> depRoots;
  // semantics initializes the predeclared Module semantics
  private final StarlarkSemantics semantics;

  // loads is a map that tracks the modules that have been loaded; it is used to
  // memoize the load graph.
  private final Map<Label, Module> loaded = new HashMap<>();
  // pending tracks the modules currently being loaded. It is used to detect
  // cycles.
  private final LinkedHashSet<Label> pending = new LinkedHashSet<>();
  // imports is a string -> module mapping that acts as the Thread loader
  // function.
  private final Map<String, Module> imports = new TreeMap();

  // --- EXPERIMENTAL ---
  // moduleGraph is a graph where nodes are modules and edges are module A ->
  // module B is A loads B. It's not really used yet, and basically duplicates
  // imports/loaded.
  private final Digraph<Module> moduleGraph = new Digraph<Module>();
  // functionCalls is a data structure that captures the functions called within
  // the body of a starlark function that receive **kwargs.
  // This is currently unused since getResolverFunction() is no longer accessible.
  private final ImmutableListMultimap.Builder<UserDefinedFunction, Collection<CallExpression>> functionCalls = ImmutableListMultimap
      .builder();

  // Missing symbols tracking for best-effort extraction
  // Maps label -> set of symbol names that were missing during evaluation
  // Used to provide stubs on retry
  private final Map<Label, LinkedHashSet<String>> missingSymbolsByLabel = new HashMap<>();

  // Native rules loaded from bundled binary protos at class initialization time
  // Maps rule name -> RuleInfo for native Bazel rules (genrule, filegroup, etc.)
  // Static to avoid reloading on every Constellate instance (worker process
  // optimization)
  private static final Map<String, RuleInfo> NATIVE_RULES = loadNativeRulesFromResources();

  // Maximum retries per file to prevent infinite loops
  private static final int MAX_RETRIES_PER_FILE = 10;

  public StarlarkEvaluator(StarlarkSemantics semantics, StarlarkFileAccessor fileAccessor,
      List<String> depRoots) {
    this.semantics = semantics;
    this.fileAccessor = fileAccessor;

    if (depRoots.isEmpty()) {
      // For backwards compatibility, if no dep_roots are specified, use the current
      // directory as the only root.
      this.depRoots = ImmutableList.of(".");
    } else {
      this.depRoots = depRoots;
    }
  }

  static boolean validSymbolName(ImmutableSet<String> symbolNames, String symbolName) {
    if (symbolNames.isEmpty()) {
      // Symbols prefixed with an underscore are private, and thus, by default,
      // documentation should not be generated for them.
      return !symbolName.startsWith("_");
    } else if (symbolNames.contains(symbolName)) {
      return true;
    } else if (symbolName.contains(".")) {
      return symbolNames.contains(symbolName.substring(0, symbolName.indexOf('.')));
    }
    return false;
  }

  /**
   * Evaluates/interprets the Starlark file at a given path and its transitive
   * Starlark dependencies using a fake build API and collects information about
   * all rule definitions made in the root Starlark file.
   *
   * @param label                  the label of the Starlark file to evaluate
   * @param ruleInfoMap            a map builder to be populated with rule
   *                               definition information for named rules. Keys
   *                               are exported names of rules, and values are
   *                               their {@link RuleInfo} rule descriptions. For
   *                               example, 'my_rule = rule(...)' has key
   *                               'my_rule'
   * @param providerInfoMap        a map builder to be populated with provider
   *                               definition information for named providers.
   *                               Keys are exported names of providers, and
   *                               values are their {@link ProviderInfo}
   *                               descriptions. For example, 'my_provider =
   *                               provider(...)' has key 'my_provider'
   * @param userDefinedFunctionMap a map builder to be populated with user-defined
   *                               functions. Keys are exported names of
   *                               functions, and values are the
   *                               {@link StarlarkFunction} objects. For example,
   *                               'def my_function(foo):' is a function with key
   *                               'my_function'.
   * @param aspectInfoMap          a map builder to be populated with aspect
   *                               definition information for named aspects. Keys
   *                               are exported names of aspects, and values are
   *                               the {@link AspectInfo} asepct descriptions. For
   *                               example, 'my_aspect = aspect(...)' has key
   *                               'my_aspect'
   * @param moduleDocMap           a map builder to be populated with module
   *                               docstrings for Starlark file. Keys are labels
   *                               of Starlark files and values are their module
   *                               docstrings. If the module has no docstring, an
   *                               empty string will be printed.
   * @throws InterruptedException if evaluation is interrupted
   */
  public Module eval(ParserInput input, Label label, ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableMap.Builder<String, ProviderInfo> providerInfoMap,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap,
      ImmutableMap.Builder<String, AspectInfo> aspectInfoMap,
      ImmutableMap.Builder<String, RepositoryRuleInfo> repositoryRuleInfoMap,
      ImmutableMap.Builder<String, ModuleExtensionInfo> moduleExtensionInfoMap,
      ImmutableMap.Builder<String, MacroInfo> macroInfoMap,
      ImmutableMap.Builder<Label, String> moduleDocMap,
      StarlarkProtos.Module.Builder starlarkModule,
      ImmutableMap.Builder<Label, Map<String, Object>> globals)
      throws InterruptedException, IOException, LabelSyntaxException, EvalException, StarlarkEvaluationException {

    // A mapping from module to the modules it loads
    // ImmutableListMultimap.Builder<Module, Collection<Module>> loadGraph =
    // ImmutableListMultimap.builder();

    List<RuleInfoWrapper> ruleInfoList = new ArrayList<>();

    List<ProviderInfoWrapper> providerInfoList = new ArrayList<>();

    List<AspectInfoWrapper> aspectInfoList = new ArrayList<>();

    List<MacroInfoWrapper> macroInfoList = new ArrayList<>();

    List<RepositoryRuleInfoWrapper> repositoryRuleInfoList = new ArrayList<>();

    List<ModuleExtensionInfoWrapper> moduleExtensionInfoList = new ArrayList<>();

    Module module = recursiveEval(input, label, ruleInfoList, providerInfoList, aspectInfoList, macroInfoList,
        repositoryRuleInfoList, moduleExtensionInfoList, moduleDocMap);

    // Extract load statements from the main file and add to proto
    StarlarkFile file = StarlarkFile.parse(input, FileOptions.DEFAULT);
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        LoadStatement loadStmt = (LoadStatement) stmt;

        // Build the LoadStmt proto
        StarlarkProtos.LoadStmt.Builder loadStmtBuilder = StarlarkProtos.LoadStmt.newBuilder();

        // Parse the load label and convert to proto Label message
        String loadLabelStr = loadStmt.getImport().getValue();
        try {
          Label parsedLabel;
          if (loadLabelStr.startsWith("//") || loadLabelStr.startsWith("@")) {
            parsedLabel = Label.parseCanonical(loadLabelStr);
          } else {
            // Relative label - parse relative to current package
            parsedLabel = Label.parseCanonical("//" + label.getPackageName() + ":" + loadLabelStr);
          }

          String repoName = parsedLabel.getRepository().getName();
          String pkgName = parsedLabel.getPackageName();
          String targetName = parsedLabel.getName();

          logger.atFine().log("Parsed load label '%s' -> repo='%s', pkg='%s', name='%s'",
              loadLabelStr, repoName, pkgName, targetName);

          StarlarkProtos.Label protoLabel = StarlarkProtos.Label.newBuilder()
              .setRepo(repoName)
              .setPkg(pkgName)
              .setName(targetName)
              .build();
          loadStmtBuilder.setLabel(protoLabel);
        } catch (LabelSyntaxException e) {
          // If label parsing fails, create a Label with just the name field
          logger.atWarning().withCause(e).log("Failed to parse load label: %s", loadLabelStr);
          loadStmtBuilder.setLabel(StarlarkProtos.Label.newBuilder()
              .setName(loadLabelStr)
              .build());
        }

        // Add location for the load statement itself
        Location loadLoc = loadStmt.getStartLocation();
        Location loadEndLoc = loadStmt.getEndLocation();
        StarlarkProtos.SymbolLocation loadSymbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName("load")
            .setStart(
                StarlarkProtos.Position.newBuilder().setLine(loadLoc.line()).setCharacter(loadLoc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loadEndLoc.line()).setCharacter(loadEndLoc.column())
                .build())
            .build();
        loadStmtBuilder.setLocation(loadSymbolLocation);

        // Add each loaded symbol with its location
        for (LoadStatement.Binding binding : loadStmt.getBindings()) {
          Location bindingLoc = binding.getLocalName().getStartLocation();
          Location bindingEndLoc = binding.getLocalName().getEndLocation();
          StarlarkProtos.LoadSymbol loadSymbol = StarlarkProtos.LoadSymbol.newBuilder()
              .setFrom(binding.getOriginalName().getName())
              .setTo(binding.getLocalName().getName())
              .setLocation(StarlarkProtos.SymbolLocation.newBuilder()
                  .setName(binding.getLocalName().getName())
                  .setStart(StarlarkProtos.Position.newBuilder().setLine(bindingLoc.line())
                      .setCharacter(bindingLoc.column()).build())
                  .setEnd(StarlarkProtos.Position.newBuilder().setLine(bindingEndLoc.line())
                      .setCharacter(bindingEndLoc.column()).build())
                  .build())
              .build();
          loadStmtBuilder.addSymbol(loadSymbol);
        }

        starlarkModule.addLoad(loadStmtBuilder.build());
      }
    }

    logger.atFine().log("\n\nresolving module globals: %s", label);

    resolveGlobals(
        module,
        label,
        ruleInfoMap,
        providerInfoMap,
        userDefinedFunctionMap,
        aspectInfoMap,
        repositoryRuleInfoMap,
        moduleExtensionInfoMap,
        macroInfoMap,
        moduleDocMap,
        ruleInfoList,
        providerInfoList,
        aspectInfoList,
        macroInfoList,
        repositoryRuleInfoList,
        moduleExtensionInfoList,
        starlarkModule);

    logger.atFine().log("post-eval rules: %s", ruleInfoMap.build().keySet());

    return module;
  }

  public void resolveGlobals(Module module,
      Label label,
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableMap.Builder<String, ProviderInfo> providerInfoMap,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap,
      ImmutableMap.Builder<String, AspectInfo> aspectInfoMap,
      ImmutableMap.Builder<String, RepositoryRuleInfo> repositoryRuleInfoMap,
      ImmutableMap.Builder<String, ModuleExtensionInfo> moduleExtensionInfoMap,
      ImmutableMap.Builder<String, MacroInfo> macroInfoMap,
      ImmutableMap.Builder<Label, String> moduleDocMap,
      List<RuleInfoWrapper> ruleInfoList,
      List<ProviderInfoWrapper> providerInfoList,
      List<AspectInfoWrapper> aspectInfoList,
      List<MacroInfoWrapper> macroInfoList,
      List<RepositoryRuleInfoWrapper> repositoryRuleInfoList,
      List<ModuleExtensionInfoWrapper> moduleExtensionInfoList,
      StarlarkProtos.Module.Builder starlarkModule)
      throws InterruptedException, IOException, LabelSyntaxException, EvalException, StarlarkEvaluationException {

    Map<StarlarkCallable, RuleInfoWrapper> ruleFunctions = ruleInfoList.stream()
        .collect(Collectors.toMap(RuleInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<StarlarkCallable, ProviderInfoWrapper> providerInfos = providerInfoList.stream()
        .collect(Collectors.toMap(ProviderInfoWrapper::getIdentifier, Functions.identity()));

    Map<StarlarkCallable, AspectInfoWrapper> aspectFunctions = aspectInfoList.stream()
        .collect(Collectors.toMap(AspectInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<StarlarkCallable, MacroInfoWrapper> macroFunctions = macroInfoList.stream()
        .collect(Collectors.toMap(MacroInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<StarlarkCallable, RepositoryRuleInfoWrapper> repositoryRuleFunctions = repositoryRuleInfoList.stream()
        .collect(Collectors.toMap(RepositoryRuleInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<Object, ModuleExtensionInfoWrapper> moduleExtensionObjects = moduleExtensionInfoList.stream()
        .collect(Collectors.toMap(ModuleExtensionInfoWrapper::getIdentifierObject, Functions.identity()));

    // Sort the globals bindings by name.
    TreeMap<String, Object> sortedBindings = new TreeMap<>(module.getGlobals());

    // calledWithKwargs represents a function identifier that was called using
    // kwargs from a user defined function. For example, if the body of `def
    // _buildifier(**kwargs)` calls `buildifier(**kwargs)`, the we store a
    // mapping [buildifier<String>, _buildifier<string>]. There can be multiple
    // alternate rules called by a macro (e.g. go_transition_wrapper=go_test and
    // go_transition_wrapper=go_binary)
    ImmutableListMultimap.Builder<String, Collection<String>> calledWithKwargs = ImmutableListMultimap.builder();

    // calledWithName tracks rules/macros that receive the 'name' parameter from a
    // function.
    // This provides stronger signal for wrapper function detection. For example, if
    // `def my_macro(name, **kwargs): my_rule(name=name, **kwargs)`, we store
    // mapping [my_rule<String>, my_macro<String>].
    ImmutableListMultimap.Builder<String, Collection<String>> calledWithName = ImmutableListMultimap.builder();

    // Log all exported symbols
    for (Entry<String, Module> loadedModule : imports.entrySet()) {
      TreeMap<String, Object> exports = new TreeMap<>(loadedModule.getValue().getGlobals());
      for (Entry<String, Object> export : exports.entrySet()) {
        logger.atFine().log("%s top-level symbol %s -> %s",
            loadedModule.getKey(), export.getKey(),
            export.getValue().getClass().getName());
      }
    }
    logger.atFine().log(
        "resolving module %s: rules: %s, providers: %s, aspects: %s, macros: %s, repository_rules: %s, module_extensions: %s",
        module.getClientData(),
        ruleInfoList.size(), providerInfoList.size(),
        aspectInfoList.size(), macroInfoList.size(), repositoryRuleInfoList.size(), moduleExtensionInfoList.size());

    for (Entry<String, Object> envEntry : sortedBindings.entrySet()) {
      logger.atFine().log("global object %s %s", envEntry.getKey(), envEntry.getValue().getClass().getName());
      // if (!envEntry.getKey().startsWith("_")) {
      // starlarkModule.addGlobal(asBinding(envEntry));
      // }

      // +++ RULES
      if (ruleFunctions.containsKey(envEntry.getValue())) {
        RuleInfoWrapper wrapper = ruleFunctions.get(envEntry.getValue());
        RuleInfo.Builder ruleInfoBuilder = wrapper.getRuleInfo();

        // Use symbol name as the rule name only if not already set in the call to
        // rule().
        if ("".equals(ruleInfoBuilder.getRuleName())) {
          ruleInfoBuilder.setRuleName(envEntry.getKey());
        }

        // Set OriginKey with the exported name and file label
        OriginKey originKey = OriginKey.newBuilder()
            .setName(envEntry.getKey())
            .setFile(label.getCanonicalForm())
            .build();
        ruleInfoBuilder.setOriginKey(originKey);

        RuleInfo ruleInfo = ruleInfoBuilder.build();
        Location loc = wrapper.getLocation();
        logger.atFine().log("global rule %s from %s", ruleInfo.getRuleName(), label);
        try {
          ruleInfoMap.put(ruleInfo.getRuleName(), ruleInfo);
        } catch (IllegalArgumentException e) {
          // ImmutableMap.Builder throws IllegalArgumentException on duplicate keys
          // Log the duplicate and skip it (use the first definition)
          logger.atWarning().log("Duplicate rule definition for '%s' in %s (keeping first definition). Error: %s",
              ruleInfo.getRuleName(), label, e.getMessage());
        }
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(ruleInfo.getRuleName())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }

      // +++ PROVIDERS
      // Handle provider(init=...) which returns (provider, raw_constructor) tuple
      Object providerValue = envEntry.getValue();
      if (providerValue instanceof Tuple && ((Tuple) providerValue).size() == 2
          && ((Tuple) providerValue).get(0) instanceof FakeProviderApi) {
        providerValue = ((Tuple) providerValue).get(0);
      }

      if (providerInfos.containsKey(providerValue)) {
        ProviderInfoWrapper wrapper = providerInfos.get(providerValue);
        ProviderInfo.Builder providerInfoBuild = wrapper.getProviderInfo();
        providerInfoBuild.setProviderName(envEntry.getKey());

        // Set OriginKey with the exported name and file label
        OriginKey originKey = OriginKey.newBuilder()
            .setName(envEntry.getKey())
            .setFile(label.getCanonicalForm())
            .build();
        providerInfoBuild.setOriginKey(originKey);

        ProviderInfo providerInfo = providerInfoBuild.build();
        Location loc = wrapper.getLocation();
        logger.atFine().log("global provider %s from %s", envEntry.getKey(), label);
        try {
          providerInfoMap.put(envEntry.getKey(), providerInfo);
        } catch (IllegalArgumentException e) {
          logger.atWarning().log("Duplicate provider definition for '%s' in %s (keeping first definition). Error: %s",
              envEntry.getKey(), label, e.getMessage());
        }
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(envEntry.getKey())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }

      // +++ FUNCTIONS
      if (envEntry.getValue() instanceof StarlarkFunction) {
        StarlarkFunction userDefinedFunction = (StarlarkFunction) envEntry.getValue();
        logger.atFine().log("global function %s", envEntry.getKey());

        if (userDefinedFunction.hasKwargs()) {
          resolveFunctionKwargs(module, envEntry.getKey(), userDefinedFunction, calledWithKwargs, calledWithName);
        } else {
          // Even without **kwargs, we should track name parameter forwarding
          resolveFunctionNameForwarding(module, envEntry.getKey(), userDefinedFunction, calledWithName);
        }
        userDefinedFunctionMap.put(envEntry.getKey(), userDefinedFunction);

        // Add symbol location for function
        Location loc = userDefinedFunction.getLocation();
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(envEntry.getKey())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }

      // +++ STRUCTS
      if (envEntry.getValue() instanceof FakeStructApi) {
        String namespaceName = envEntry.getKey();
        FakeStructApi namespace = (FakeStructApi) envEntry.getValue();
        logger.atFine().log("global struct %s.%s", namespaceName, namespace);
        putStructFields(namespaceName, namespace, userDefinedFunctionMap);
      }

      // +++ GLOBAL SCALARS (string, int, bool, list)
      // Only capture public symbols (not starting with _) to reduce proto size
      if (!envEntry.getKey().startsWith("_")) {
        StarlarkProtos.Value value = convertToValue(envEntry.getValue(), envEntry.getKey());
        if (value != null) {
          starlarkModule.putGlobal(envEntry.getKey(), value);
        }
      }

      // +++ ASPECTS
      if (aspectFunctions.containsKey(envEntry.getValue())) {
        AspectInfoWrapper wrapper = aspectFunctions.get(envEntry.getValue());
        AspectInfo.Builder aspectInfoBuild = wrapper.getAspectInfo();
        aspectInfoBuild.setAspectName(envEntry.getKey());

        // Set OriginKey with the exported name and file label
        OriginKey originKey = OriginKey.newBuilder()
            .setName(envEntry.getKey())
            .setFile(label.getCanonicalForm())
            .build();
        aspectInfoBuild.setOriginKey(originKey);

        AspectInfo aspectInfo = aspectInfoBuild.build();
        logger.atFine().log("global aspect %s from %s", envEntry.getKey(), label);
        try {
          aspectInfoMap.put(envEntry.getKey(), aspectInfo);
        } catch (IllegalArgumentException e) {
          logger.atWarning().log("Duplicate aspect definition for '%s' in %s (keeping first definition). Error: %s",
              envEntry.getKey(), label, e.getMessage());
        }

        // Add symbol location for aspect
        Location loc = wrapper.getLocation();
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(envEntry.getKey())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }

      // +++ MACROS
      if (macroFunctions.containsKey(envEntry.getValue())) {
        MacroInfoWrapper wrapper = macroFunctions.get(envEntry.getValue());
        MacroInfo.Builder macroInfoBuild = wrapper.getMacroInfo();
        macroInfoBuild.setMacroName(envEntry.getKey());

        // Set OriginKey with the exported name and file label
        OriginKey originKey = OriginKey.newBuilder()
            .setName(envEntry.getKey())
            .setFile(label.getCanonicalForm())
            .build();
        macroInfoBuild.setOriginKey(originKey);

        MacroInfo macroInfo = macroInfoBuild.build();
        Location loc = wrapper.getLocation();
        logger.atFine().log("global macro %s from %s", envEntry.getKey(), label);
        try {
          macroInfoMap.put(envEntry.getKey(), macroInfo);
        } catch (IllegalArgumentException e) {
          logger.atWarning().log("Duplicate macro definition for '%s' in %s (keeping first definition). Error: %s",
              envEntry.getKey(), label, e.getMessage());
        }
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(macroInfo.getMacroName())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }

      // +++ REPOSITORY RULES
      if (repositoryRuleFunctions.containsKey(envEntry.getValue())) {
        RepositoryRuleInfoWrapper wrapper = repositoryRuleFunctions.get(envEntry.getValue());
        RepositoryRuleInfo.Builder repositoryRuleInfoBuild = wrapper.getRepositoryRuleInfo();
        repositoryRuleInfoBuild.setRuleName(envEntry.getKey());

        // Set OriginKey with the exported name and file label
        OriginKey originKey = OriginKey.newBuilder()
            .setName(envEntry.getKey())
            .setFile(label.getCanonicalForm())
            .build();
        repositoryRuleInfoBuild.setOriginKey(originKey);

        RepositoryRuleInfo repositoryRuleInfo = repositoryRuleInfoBuild.build();
        Location loc = wrapper.getLocation();
        logger.atFine().log("global repository_rule %s from %s", envEntry.getKey(), label);
        try {
          repositoryRuleInfoMap.put(envEntry.getKey(), repositoryRuleInfo);
        } catch (IllegalArgumentException e) {
          logger.atWarning().log(
              "Duplicate repository_rule definition for '%s' in %s (keeping first definition). Error: %s",
              envEntry.getKey(), label, e.getMessage());
        }
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(repositoryRuleInfo.getRuleName())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }

      // +++ MODULE EXTENSIONS
      if (moduleExtensionObjects.containsKey(envEntry.getValue())) {
        ModuleExtensionInfoWrapper wrapper = moduleExtensionObjects.get(envEntry.getValue());
        ModuleExtensionInfo.Builder moduleExtensionInfoBuild = wrapper.getModuleExtensionInfo();
        moduleExtensionInfoBuild.setExtensionName(envEntry.getKey());

        // Set OriginKey with the exported name and file label
        OriginKey originKey = OriginKey.newBuilder()
            .setName(envEntry.getKey())
            .setFile(label.getCanonicalForm())
            .build();
        moduleExtensionInfoBuild.setOriginKey(originKey);

        ModuleExtensionInfo moduleExtensionInfo = moduleExtensionInfoBuild.build();
        Location loc = wrapper.getLocation();
        logger.atFine().log("global module_extension %s from %s", envEntry.getKey(), label);
        try {
          moduleExtensionInfoMap.put(envEntry.getKey(), moduleExtensionInfo);
        } catch (IllegalArgumentException e) {
          logger.atWarning().log(
              "Duplicate module_extension definition for '%s' in %s (keeping first definition). Error: %s",
              envEntry.getKey(), label, e.getMessage());
        }
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(moduleExtensionInfo.getExtensionName())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }
    }

    // iterate all the rules and see if there is a corresponding macro that
    // calls it. For example, we might encounter a RuleInfo for '_buildifier'.
    // if there is a corresponding macro userDefinedFunction that called it via
    // passing it's kwargs, add an entry to the rule Map
    logger.atFine().log("attempting to resolve %d macros: %s", calledWithKwargs.build().size(), calledWithKwargs);

    // might be better to iterate the list of userDefinedFunctions and see if it
    // called something with kwargs.
    // resolveMacrosRule(ruleInfoMap, ruleInfoList, calledWithKwargs.build(),
    // userDefinedFunctionMap.build());
    resolveFunctionMacros(ruleInfoMap, ruleInfoList, calledWithKwargs.build(), userDefinedFunctionMap.build());

    // Add rules from loaded modules to the rule map
    // This allows detection of RuleMacros that forward to rules from other files
    Set<String> currentRuleNames = new HashSet<>();
    // Collect current rule names to avoid duplicates
    for (RuleInfoWrapper wrapper : ruleInfoList) {
      if (wrapper.getIdentifierFunction() instanceof PostAssignHookAssignableIdentifier) {
        PostAssignHookAssignableIdentifier ident = (PostAssignHookAssignableIdentifier) wrapper.getIdentifierFunction();
        currentRuleNames.add(ident.getAssignedName());
      }
    }

    for (Module loadedModule : imports.values()) {
      for (Map.Entry<String, Object> entry : loadedModule.getGlobals().entrySet()) {
        if (entry.getValue() instanceof FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) {
          FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier ruleIdent = (FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) entry
              .getValue();
          String ruleName = entry.getKey();

          // Skip if this rule is already in our current module
          if (currentRuleNames.contains(ruleName)) {
            continue;
          }

          // Check if we have RuleInfo for this rule
          // Look through ruleInfoList for a matching rule by the identifier
          for (RuleInfoWrapper wrapper : ruleInfoList) {
            if (wrapper.getIdentifierFunction() instanceof PostAssignHookAssignableIdentifier) {
              PostAssignHookAssignableIdentifier ident = (PostAssignHookAssignableIdentifier) wrapper
                  .getIdentifierFunction();
              if (ident.getAssignedName().equals(ruleName)) {
                RuleInfo ruleInfo = wrapper.getRuleInfo().build();
                try {
                  ruleInfoMap.put(ruleName, ruleInfo);
                  logger.atFine().log("Added rule '%s' from loaded module to rule map", ruleName);
                } catch (IllegalArgumentException e) {
                  logger.atFine().log("Rule '%s' already in map, skipping", ruleName);
                }
                break;
              }
            }
          }
        }
      }
    }

    // Second pass: analyze functions to detect calls to rules/aspects/macros
    // (wrapper functions)
    // This must be done after all rules/aspects/macros have been added to their
    // respective maps
    ImmutableMap<String, RuleInfo> builtRuleInfoMap = ruleInfoMap.build();
    ImmutableMap<String, AspectInfo> builtAspectInfoMap = aspectInfoMap.build();
    ImmutableMap<String, MacroInfo> builtMacroInfoMap = macroInfoMap.build();
    ImmutableMap<String, StarlarkFunction> builtUserFunctionMap = userDefinedFunctionMap.build();

    // Build inverted map: functionName -> [rules/macros that receive **kwargs from
    // this function]
    // calledWithKwargs maps ruleName -> [functionNames], we need to invert it
    ImmutableListMultimap<String, Collection<String>> builtCalledWithKwargs = calledWithKwargs.build();
    Map<String, List<String>> functionToKwargsTargets = new HashMap<>();
    for (Entry<String, Collection<String>> entry : builtCalledWithKwargs.entries()) {
      String ruleName = entry.getKey();
      for (String functionName : entry.getValue()) {
        functionToKwargsTargets.computeIfAbsent(functionName, k -> new ArrayList<>()).add(ruleName);
      }
    }

    // Build inverted map: functionName -> [rules/macros that receive 'name'
    // parameter from this function]
    // calledWithName maps ruleName -> [functionNames], we need to invert it
    ImmutableListMultimap<String, Collection<String>> builtCalledWithName = calledWithName.build();
    Map<String, List<String>> functionToNameTargets = new HashMap<>();
    for (Entry<String, Collection<String>> entry : builtCalledWithName.entries()) {
      String ruleName = entry.getKey();
      for (String functionName : entry.getValue()) {
        functionToNameTargets.computeIfAbsent(functionName, k -> new ArrayList<>()).add(ruleName);
      }
    }

    // Track functions that should be converted to RuleMacros
    Set<String> ruleMacroFunctionNames = new HashSet<>();

    for (Entry<String, StarlarkFunction> funcEntry : builtUserFunctionMap.entrySet()) {
      StarlarkFunction userDefinedFunction = funcEntry.getValue();
      String functionName = funcEntry.getKey();

      // Skip private functions (those starting with underscore)
      if (functionName.startsWith("_")) {
        continue;
      }

      // Analyze function body to detect calls to rules/aspects/macros
      List<String> calledRulesAndMacros = findRuleAndMacroCallsInFunction(
          userDefinedFunction,
          builtRuleInfoMap,
          builtAspectInfoMap,
          builtMacroInfoMap,
          ruleInfoList,
          NATIVE_RULES);

      // Get the list of rules/macros that receive **kwargs from this function
      List<String> kwargsTargets = functionToKwargsTargets.getOrDefault(functionName, ImmutableList.of());

      // Get the list of rules/macros that receive 'name' parameter from this function
      List<String> nameTargets = functionToNameTargets.getOrDefault(functionName, ImmutableList.of());

      // Check if this function should be a RuleMacro:
      // - Function forwards kwargs or name to a rule
      // - The target is a rule in our rule map OR in the ruleInfoList OR a native
      // rule
      // Note: We don't restrict to private rules (_) - real-world macros often wrap
      // public rules too
      String wrappedRuleName = null;
      for (String target : kwargsTargets) {
        if (builtRuleInfoMap.containsKey(target)) {
          wrappedRuleName = target;
          break;
        }
        // Check if this is a native.* rule
        if (target.startsWith("native.")) {
          String nativeRuleName = target.substring("native.".length());
          if (NATIVE_RULES.containsKey(nativeRuleName)) {
            wrappedRuleName = target;
            break;
          }
        }
        // Also check if there's a wrapper for this rule (might be from a loaded module)
        for (RuleInfoWrapper wrapper : ruleInfoList) {
          if (wrapper.getIdentifierFunction() instanceof PostAssignHookAssignableIdentifier) {
            PostAssignHookAssignableIdentifier ident = (PostAssignHookAssignableIdentifier) wrapper
                .getIdentifierFunction();
            if (ident.getAssignedName().equals(target)) {
              wrappedRuleName = target;
              break;
            }
          }
        }
        if (wrappedRuleName != null)
          break;
      }
      // NOTE: We don't check nameTargets here for RuleMacro classification.
      // Almost every function passes 'name' through, so that would be too broad.
      // RuleMacro is specifically for functions that forward **kwargs to rules.

      if (wrappedRuleName != null) {
        // This is a RuleMacro - convert it
        // We need to create both Function and Rule wrappers, then link them in
        // RuleMacro
        ruleMacroFunctionNames.add(functionName);
        try {
          StarlarkFunctionInfo functionInfo = FunctionUtil.fromNameAndFunction(functionName, userDefinedFunction);
          RuleInfo ruleInfo = builtRuleInfoMap.get(wrappedRuleName);
          // If not in the map, check if it's a native rule
          if (ruleInfo == null && wrappedRuleName.startsWith("native.")) {
            String nativeRuleName = wrappedRuleName.substring("native.".length());
            ruleInfo = NATIVE_RULES.get(nativeRuleName);
          }
          // If still not found, try to find it in ruleInfoList (might be from a loaded
          // module)
          if (ruleInfo == null) {
            for (RuleInfoWrapper wrapper : ruleInfoList) {
              if (wrapper.getIdentifierFunction() instanceof PostAssignHookAssignableIdentifier) {
                PostAssignHookAssignableIdentifier ident = (PostAssignHookAssignableIdentifier) wrapper
                    .getIdentifierFunction();
                if (ident.getAssignedName().equals(wrappedRuleName)) {
                  // Get the builder and ensure the rule name is set before building
                  RuleInfo.Builder ruleInfoBuilder = wrapper.getRuleInfo();
                  if ("".equals(ruleInfoBuilder.getRuleName())) {
                    ruleInfoBuilder.setRuleName(wrappedRuleName);
                  }
                  ruleInfo = ruleInfoBuilder.build();
                  break;
                }
              }
            }
          }
          if (ruleInfo == null) {
            logger.atWarning().log("Could not find RuleInfo for %s, skipping RuleMacro creation", wrappedRuleName);
            continue; // Skip this function
          }
          Location funcLoc = userDefinedFunction.getLocation();
          StarlarkProtos.SymbolLocation funcSymbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
              .setName(functionName)
              .setStart(
                  StarlarkProtos.Position.newBuilder().setLine(funcLoc.line()).setCharacter(funcLoc.column()).build())
              .setEnd(
                  StarlarkProtos.Position.newBuilder().setLine(funcLoc.line()).setCharacter(funcLoc.column()).build())
              .build();

          // Create Function wrapper
          StarlarkProtos.Function.Builder functionBuilder = StarlarkProtos.Function.newBuilder()
              .setInfo(functionInfo)
              .setLocation(funcSymbolLocation)
              .addAllCallsRuleOrMacro(calledRulesAndMacros)
              .addAllForwardsKwargsTo(kwargsTargets)
              .addAllForwardsNameTo(nameTargets);

          // Create Rule wrapper for the private rule
          StarlarkProtos.Rule.Builder ruleBuilder = StarlarkProtos.Rule.newBuilder()
              .setInfo(ruleInfo);

          // Find the rule's location from the rule wrapper list
          boolean foundRuleLocation = false;
          for (RuleInfoWrapper ruleWrapper : ruleInfoList) {
            String ruleWrapperName = ((PostAssignHookAssignableIdentifier) ruleWrapper.getIdentifierFunction())
                .getAssignedName();
            if (ruleWrapperName.equals(wrappedRuleName)) {
              Location ruleLoc = ruleWrapper.getLocation();
              StarlarkProtos.SymbolLocation ruleSymbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
                  .setName(wrappedRuleName)
                  .setStart(StarlarkProtos.Position.newBuilder().setLine(ruleLoc.line()).setCharacter(ruleLoc.column())
                      .build())
                  .setEnd(StarlarkProtos.Position.newBuilder().setLine(ruleLoc.line()).setCharacter(ruleLoc.column())
                      .build())
                  .build();
              ruleBuilder.setLocation(ruleSymbolLocation);
              foundRuleLocation = true;
              break;
            }
          }
          // For native rules, set a built-in location since they're not in ruleInfoList
          if (!foundRuleLocation && wrappedRuleName.startsWith("native.")) {
            StarlarkProtos.SymbolLocation builtinLocation = StarlarkProtos.SymbolLocation.newBuilder()
                .setName(wrappedRuleName)
                .setStart(StarlarkProtos.Position.newBuilder().setLine(0).setCharacter(0).build())
                .setEnd(StarlarkProtos.Position.newBuilder().setLine(0).setCharacter(0).build())
                .build();
            ruleBuilder.setLocation(builtinLocation);
          }

          // Add attributes from the rule
          for (AttributeInfo attrInfo : ruleInfo.getAttributeList()) {
            StarlarkProtos.Attribute.Builder attrBuilder = StarlarkProtos.Attribute.newBuilder()
                .setInfo(attrInfo);
            ruleBuilder.addAttribute(attrBuilder.build());
          }

          // Create RuleMacro linking function and rule
          StarlarkProtos.RuleMacro.Builder ruleMacroBuilder = StarlarkProtos.RuleMacro.newBuilder()
              .setFunction(functionBuilder.build())
              .setRule(ruleBuilder.build());

          // Add other symbols that the function called (not via kwargs/name forwarding)
          for (String symbol : calledRulesAndMacros) {
            if (!symbol.equals(wrappedRuleName)) {
              ruleMacroBuilder.addSymbol(symbol);
            }
          }

          starlarkModule.addRuleMacro(ruleMacroBuilder.build());
          logger.atFine().log("Converted function %s to RuleMacro wrapping %s", functionName, wrappedRuleName);
        } catch (ExtractionException e) {
          logger.atWarning().log("Could not convert function %s to RuleMacro: %s",
              functionName, e.getMessage());
        }
      } else {
        // Regular function - create Function proto
        try {
          StarlarkFunctionInfo functionInfo = FunctionUtil.fromNameAndFunction(functionName, userDefinedFunction);
          Location loc = userDefinedFunction.getLocation();
          StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
              .setName(functionName)
              .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
              .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
              .build();

          StarlarkProtos.Function.Builder functionBuilder = StarlarkProtos.Function.newBuilder()
              .setInfo(functionInfo)
              .setLocation(symbolLocation)
              .addAllCallsRuleOrMacro(calledRulesAndMacros)
              .addAllForwardsKwargsTo(kwargsTargets)
              .addAllForwardsNameTo(nameTargets);
          starlarkModule.addFunction(functionBuilder.build());
        } catch (ExtractionException e) {
          // If we can't extract function info, log and continue
          logger.atWarning().log("Could not extract function info for %s: %s",
              functionName, e.getMessage());
        }
      }
    }

  }

  private void resolveFunctionMacros(
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      List<RuleInfoWrapper> ruleInfoList,
      ImmutableListMultimap<String, Collection<String>> rulesCalledWithKwargs,
      ImmutableMap<String, StarlarkFunction> userFunctions) {

    // Build a map from rule names to their info for lookup
    Map<String, RuleInfo> ruleNameToInfo = new HashMap<>();
    for (RuleInfoWrapper wrapper : ruleInfoList) {
      String name = ((PostAssignHookAssignableIdentifier) wrapper.getIdentifierFunction()).getAssignedName();
      ruleNameToInfo.put(name, wrapper.getRuleInfo().build());
    }

    // Track which macro names we've already added to avoid duplicates
    Set<String> addedMacros = new HashSet<>();

    for (String ruleName : rulesCalledWithKwargs.keys()) {
      if (!ruleNameToInfo.containsKey(ruleName)) {
        logger.atFine().log("resolveFunctionMacros: skipping rule %s (unknown)", ruleName);
        continue;
      }
      RuleInfo ruleInfo = ruleNameToInfo.get(ruleName);
      for (Collection<String> macroNames : rulesCalledWithKwargs.get(ruleName)) {
        for (String macroName : macroNames) {
          // Skip if we've already added this macro name
          if (addedMacros.contains(macroName)) {
            logger.atFine().log("resolveFunctionMacros: skipping duplicate macro %s", macroName);
            continue;
          }

          StarlarkFunction function = userFunctions.get(macroName);
          RuleInfo.Builder macroInfo = ruleInfo.toBuilder();
          macroInfo.setRuleName(macroName);

          try {
            StarlarkFunctionInfo functionInfo = FunctionUtil.fromNameAndFunction(macroName, function);
            // if the def had a docstring, append it in front of the rule docstring.
            if (!Strings.isNullOrEmpty(functionInfo.getDocString())) {
              String docString = functionInfo.getDocString();
              if (!Strings.isNullOrEmpty(ruleInfo.getDocString())) {
                docString += "\n\n" + ruleInfo.getDocString();
              }
              macroInfo.setDocString(docString);
            }
          } catch (ExtractionException dspex) {
            // best-effort, ignore error
          }

          ruleInfoMap.put(macroName, macroInfo.build());
          addedMacros.add(macroName);
          logger.atFine().log("global macro %s (rule from %s, called by function %s)", macroName, ruleName,
              function.getName());
        }
      }
    }
  }

  private void resolveMacrosRule(
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      List<RuleInfoWrapper> ruleInfoList,
      ImmutableListMultimap<String, Collection<String>> rulesCalledWithKwargs,
      ImmutableMap<String, StarlarkFunction> userFunctions) {
    // iterate the list of rules. If a rule was called by a function with
    // kwargs, create a new entry in the ruleInfoMap with a copy of the ruleInfo
    // under the caller function name.
    for (RuleInfoWrapper ruleWrapper : ruleInfoList) {
      String name = ((PostAssignHookAssignableIdentifier) ruleWrapper.getIdentifierFunction()).getAssignedName();
      if (!rulesCalledWithKwargs.containsKey(name)) {
        continue;
      }
      for (Collection<String> callers : rulesCalledWithKwargs.get(name)) {
        for (String caller : callers) {
          // avoid duplicate entries
          if (ruleInfoMap.build().containsKey(caller)) {
            continue;
          }

          // Maybe improve the docstring
          StarlarkFunction function = userFunctions.get(caller);
          RuleInfo.Builder ruleInfo = ruleWrapper.getRuleInfo().clone();
          ruleInfo.setRuleName(caller);
          try {
            StarlarkFunctionInfo functionInfo = FunctionUtil.fromNameAndFunction(caller, function);
            // if the def had a docstring, append it in front of the rule docstring.
            if (!Strings.isNullOrEmpty(functionInfo.getDocString())) {
              String docString = functionInfo.getDocString();
              if (!Strings.isNullOrEmpty(ruleInfo.getDocString())) {
                docString += "\n\n" + ruleInfo.getDocString();
              }
              ruleInfo.setDocString(docString);
            }
          } catch (ExtractionException dspex) {
            // best-effort, ignore error
          }

          try {
            ruleInfoMap.put(caller, ruleInfo.build());
          } catch (IllegalArgumentException e) {
            logger.atWarning().log(
                "Duplicate macro/rule definition for '%s' (from rule %s via %s) (keeping first definition). Error: %s",
                caller, name, function.getName(), e.getMessage());
          }
          logger.atFine().log("global macro %s (rule from %s, called by %s)", caller, name, function.getName());
        }
      }
    }
  }

  static void resolveFunctionKwargs(
      Module module,
      String globalName,
      StarlarkFunction userDefinedFunction,
      ImmutableListMultimap.Builder<String, Collection<String>> calledWithKwargs,
      ImmutableListMultimap.Builder<String, Collection<String>> calledWithName) {
    logger.atFine().log("** global function %s has kwargs", globalName);

    // Check if function has a 'name' parameter
    List<String> paramNames = userDefinedFunction.getParameterNames();
    boolean hasNameParam = paramNames.contains("name");

    NodeVisitor checker = new NodeVisitor() {
      Stack<Node> stack = new Stack<>();

      @Override
      public void visit(Node node) {
        stack.push(node);
        super.visit(node);
        stack.pop();
      }

      // Record f(*args) and f(**kwargs) calls, and name parameter forwarding.
      void recordStarArgs(CallExpression call) {
        for (Argument arg : call.getArguments()) {
          if (arg instanceof Argument.StarStar) {
            logger.atFine().log("Found **kwargs forwarding at %s: %s", arg.getStartLocation(), arg.getValue());
            for (int i = stack.size() - 1; i >= 0; i--) {
              Node parent = stack.get(i);
              if (parent instanceof CallExpression) {
                CallExpression parentCall = (CallExpression) parent;
                Expression parentCallExpr = parentCall.getFunction();
                if (parentCallExpr instanceof Identifier) {
                  Identifier ident = (Identifier) parentCallExpr;
                  try {
                    Object resolved = resolveFunctionIdentifier(userDefinedFunction, ident);
                    if (resolved instanceof FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) {
                      FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier ruleIdent = (FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) resolved;
                      logger.atFine().log("Function %s forwards **kwargs to rule %s", globalName,
                          ruleIdent.getAssignedName());
                      calledWithKwargs.put(ruleIdent.getAssignedName(), ImmutableList.of(globalName));
                    } else if (resolved != null) {
                      logger.atFine().log("Function %s forwards **kwargs to %s", globalName, ident.getName());
                      calledWithKwargs.put(ident.getName(), ImmutableList.of(globalName));
                    } else {
                      // Couldn't resolve the identifier (likely from a loaded module)
                      // Add it by name and hope we can match it later
                      logger.atFine().log("Function %s forwards **kwargs to unresolved symbol %s (likely loaded)",
                          globalName, ident.getName());
                      calledWithKwargs.put(ident.getName(), ImmutableList.of(globalName));
                    }
                  } catch (InterruptedException iEx) {
                    logger.atFine().log("  ** parent-call-expression interrupt exception: %s", iEx);
                  } catch (EvalException evalEx) {
                    logger.atFine().log("  ** parent-call-expression eval exception: %s", evalEx);
                  } catch (Exception e) {
                    logger.atFine().log("  ** parent-call-expression exception: %s", e);
                  }
                } else if (parentCallExpr instanceof DotExpression) {
                  // Handle native.* calls (e.g., native.genrule(**kwargs))
                  DotExpression dotExpr = (DotExpression) parentCallExpr;
                  Expression object = dotExpr.getObject();
                  String field = dotExpr.getField().getName();
                  if (object instanceof Identifier && ((Identifier) object).getName().equals("native")) {
                    String nativeCallName = "native." + field;
                    logger.atFine().log("Function %s forwards **kwargs to %s", globalName, nativeCallName);
                    calledWithKwargs.put(nativeCallName, ImmutableList.of(globalName));
                  }
                }
                break;
              }
            }
          } else if (arg instanceof Argument.Star) {
            logger.atFine().log("STAR %s: %s", arg.getStartLocation(), arg.getValue());
          } else if (arg instanceof Argument.Keyword) {
            // Check if this is a 'name' parameter being forwarded
            Argument.Keyword kwArg = (Argument.Keyword) arg;
            if (kwArg.getName().equals("name") && hasNameParam) {
              Expression value = kwArg.getValue();
              // Check if the value references the 'name' parameter
              if (referencesNameParameter(value)) {
                logger.atFine().log("Found name parameter forwarding at %s: %s", arg.getStartLocation(), value);
                // Find the parent call expression to identify which rule/macro receives the
                // name
                for (int i = stack.size() - 1; i >= 0; i--) {
                  Node parent = stack.get(i);
                  if (parent instanceof CallExpression) {
                    CallExpression parentCall = (CallExpression) parent;
                    Expression parentCallExpr = parentCall.getFunction();
                    if (parentCallExpr instanceof Identifier) {
                      Identifier ident = (Identifier) parentCallExpr;
                      try {
                        Object resolved = resolveFunctionIdentifier(userDefinedFunction, ident);
                        if (resolved instanceof FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) {
                          FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier ruleIdent = (FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) resolved;
                          logger.atFine().log("Function %s forwards name to rule %s", globalName,
                              ruleIdent.getAssignedName());
                          calledWithName.put(ruleIdent.getAssignedName(), ImmutableList.of(globalName));
                        } else if (resolved != null) {
                          logger.atFine().log("Function %s forwards name to %s", globalName, ident.getName());
                          calledWithName.put(ident.getName(), ImmutableList.of(globalName));
                        } else {
                          // Couldn't resolve the identifier (likely from a loaded module)
                          logger.atFine().log("Function %s forwards name to unresolved symbol %s (likely loaded)",
                              globalName, ident.getName());
                          calledWithName.put(ident.getName(), ImmutableList.of(globalName));
                        }
                      } catch (InterruptedException iEx) {
                        logger.atFine().log("  ** parent-call-expression interrupt exception: %s", iEx);
                      } catch (EvalException evalEx) {
                        logger.atFine().log("  ** parent-call-expression eval exception: %s", evalEx);
                      } catch (Exception e) {
                        logger.atFine().log("  ** parent-call-expression exception: %s", e);
                      }
                    } else if (parentCallExpr instanceof DotExpression) {
                      // Handle native.* calls (e.g., native.genrule(name = name))
                      DotExpression dotExpr = (DotExpression) parentCallExpr;
                      Expression object = dotExpr.getObject();
                      String field = dotExpr.getField().getName();
                      if (object instanceof Identifier && ((Identifier) object).getName().equals("native")) {
                        String nativeCallName = "native." + field;
                        logger.atFine().log("Function %s forwards name to %s", globalName, nativeCallName);
                        calledWithName.put(nativeCallName, ImmutableList.of(globalName));
                      }
                    }
                    break;
                  }
                }
              }
            }
          }
        }
      }

      /**
       * Checks if an expression references the 'name' parameter.
       * This includes direct references (name) and expressions using name (name +
       * "_suffix").
       */
      boolean referencesNameParameter(Expression expr) {
        if (expr instanceof Identifier) {
          return ((Identifier) expr).getName().equals("name");
        } else if (expr instanceof net.starlark.java.syntax.BinaryOperatorExpression) {
          // Check if either operand references name (e.g., name + "_suffix")
          net.starlark.java.syntax.BinaryOperatorExpression binOp = (net.starlark.java.syntax.BinaryOperatorExpression) expr;
          return referencesNameParameter(binOp.getX()) || referencesNameParameter(binOp.getY());
        }
        // Could extend to other expression types if needed
        return false;
      }

      @Override
      public void visit(CallExpression node) {
        recordStarArgs(node);
        super.visit(node);
      }
    };

    // Use the public getResolverFunction() method we added to StarlarkFunction
    // to access the function body for kwargs analysis
    checker.visitAll(userDefinedFunction.getResolverFunction().getBody());
  }

  /**
   * Analyzes a user-defined function to detect name parameter forwarding
   * (for functions that don't use **kwargs).
   */
  static void resolveFunctionNameForwarding(
      Module module,
      String globalName,
      StarlarkFunction userDefinedFunction,
      ImmutableListMultimap.Builder<String, Collection<String>> calledWithName) {
    logger.atFine().log("** analyzing function %s for name parameter forwarding", globalName);

    // Check if function has a 'name' parameter
    List<String> paramNames = userDefinedFunction.getParameterNames();
    boolean hasNameParam = paramNames.contains("name");

    if (!hasNameParam) {
      return; // No name parameter to track
    }

    NodeVisitor checker = new NodeVisitor() {
      Stack<Node> stack = new Stack<>();

      @Override
      public void visit(Node node) {
        stack.push(node);
        super.visit(node);
        stack.pop();
      }

      // Check for name parameter forwarding in call expressions
      void checkNameForwarding(CallExpression call) {
        Expression callTarget = call.getFunction();

        // Check each argument to see if 'name' is being forwarded
        boolean forwardsName = false;
        for (Argument arg : call.getArguments()) {
          if (arg instanceof Argument.Keyword) {
            Argument.Keyword kwArg = (Argument.Keyword) arg;
            if (kwArg.getName().equals("name")) {
              Expression value = kwArg.getValue();
              if (referencesNameParameter(value)) {
                forwardsName = true;
                break;
              }
            }
          } else if (arg instanceof Argument.Positional) {
            // Check if the first positional argument is 'name'
            // This is less common but valid: my_rule(name, srcs=...)
            Expression value = ((Argument.Positional) arg).getValue();
            if (value instanceof Identifier && ((Identifier) value).getName().equals("name")) {
              forwardsName = true;
              break;
            }
          }
        }

        if (!forwardsName) {
          return;
        }

        // Handle Identifier calls (regular rules/macros)
        if (callTarget instanceof Identifier) {
          Identifier targetIdent = (Identifier) callTarget;
          logger.atFine().log("Found name parameter forwarding to %s at %s", targetIdent.getName(),
              call.getStartLocation());
          try {
            Object resolved = resolveFunctionIdentifier(userDefinedFunction, targetIdent);
            if (resolved instanceof FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) {
              FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier ruleIdent = (FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) resolved;
              logger.atFine().log("Function %s forwards name to rule %s", globalName, ruleIdent.getAssignedName());
              calledWithName.put(ruleIdent.getAssignedName(), ImmutableList.of(globalName));
            } else if (resolved != null) {
              logger.atFine().log("Function %s forwards name to %s", globalName, targetIdent.getName());
              calledWithName.put(targetIdent.getName(), ImmutableList.of(globalName));
            } else {
              // Couldn't resolve the identifier (likely from a loaded module)
              logger.atFine().log("Function %s forwards name to unresolved symbol %s (likely loaded)", globalName,
                  targetIdent.getName());
              calledWithName.put(targetIdent.getName(), ImmutableList.of(globalName));
            }
          } catch (Exception e) {
            logger.atFine().log("Exception resolving identifier: %s", e.getMessage());
          }
        }
        // Handle DotExpression calls (native.* rules)
        else if (callTarget instanceof DotExpression) {
          DotExpression dotExpr = (DotExpression) callTarget;
          Expression object = dotExpr.getObject();
          String field = dotExpr.getField().getName();
          if (object instanceof Identifier && ((Identifier) object).getName().equals("native")) {
            String nativeCallName = "native." + field;
            logger.atFine().log("Found name parameter forwarding to %s at %s", nativeCallName, call.getStartLocation());
            logger.atFine().log("Function %s forwards name to %s", globalName, nativeCallName);
            calledWithName.put(nativeCallName, ImmutableList.of(globalName));
          }
        }
      }

      /**
       * Checks if an expression references the 'name' parameter.
       */
      boolean referencesNameParameter(Expression expr) {
        if (expr instanceof Identifier) {
          return ((Identifier) expr).getName().equals("name");
        } else if (expr instanceof net.starlark.java.syntax.BinaryOperatorExpression) {
          net.starlark.java.syntax.BinaryOperatorExpression binOp = (net.starlark.java.syntax.BinaryOperatorExpression) expr;
          return referencesNameParameter(binOp.getX()) || referencesNameParameter(binOp.getY());
        }
        return false;
      }

      @Override
      public void visit(CallExpression node) {
        checkNameForwarding(node);
        super.visit(node);
      }
    };

    // Use the public getResolverFunction() method to access the function body
    checker.visitAll(userDefinedFunction.getResolverFunction().getBody());
  }

  static class UserDefinedFunction {
    public final Module module;
    public final StarlarkFunction function;
    public final String assignedName;

    UserDefinedFunction(Module module, StarlarkFunction function, String assignedName) {
      this.module = module;
      this.function = function;
      this.assignedName = assignedName;
    }

    @Override
    public int hashCode() {
      return Objects.hash(module, function, assignedName);
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof UserDefinedFunction)) {
        return false;
      }
      UserDefinedFunction r = (UserDefinedFunction) o;
      return module.equals(r.module) && function.equals(r.function) && assignedName == r.assignedName;
    }

  }

  /**
   * Recursively adds functions defined in {@code namespace}, and in its nested
   * namespaces, to {@code userDefinedFunctionMap}.
   *
   * <p>
   * Each entry's key is the fully qualified function name, e.g. {@code
   * "outernamespace.innernamespace.func"}. {@code namespaceName} is the fully
   * qualified name of {@code namespace} itself.
   */
  private static void putStructFields(String namespaceName, FakeStructApi namespace,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap) throws EvalException {
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
   * Recursively evaluates/interprets the Starlark file at a given path and its
   * transitive Starlark dependencies using a fake build API and collects
   * information about all rule definitions made in those files.
   *
   * @param label        the label of the Starlark file to evaluate
   * @param ruleInfoList a collection of all rule definitions made so far (using
   *                     rule()); this method will add to this list as it
   *                     evaluates additional files
   * @throws InterruptedException if evaluation is interrupted
   */
  private Module recursiveEval(ParserInput input, Label label, List<RuleInfoWrapper> ruleInfoList,
      List<ProviderInfoWrapper> providerInfoList, List<AspectInfoWrapper> aspectInfoList,
      List<MacroInfoWrapper> macroInfoList, List<RepositoryRuleInfoWrapper> repositoryRuleInfoList,
      List<ModuleExtensionInfoWrapper> moduleExtensionInfoList,
      ImmutableMap.Builder<Label, String> moduleDocMap)
      throws InterruptedException, IOException, LabelSyntaxException, StarlarkEvaluationException, EvalException {

    if (pending.contains(label)) {
      throw new StarlarkEvaluationException("cycle with " + label);
    } else if (loaded.containsKey(label)) {
      return loaded.get(label);
    }
    pending.add(label);

    logger.atFine().log("recursiveEval %s", label);

    // Create an initial environment with a fake build API. Then use Starlark's name
    // resolution
    // step to further populate the environment with all additional symbols not in
    // the fake build
    // API but used by the program; these become FakeDeepStructures.
    ImmutableMap.Builder<String, Object> initialEnvBuilder = ImmutableMap.builder();
    FakeApi.addPredeclared(initialEnvBuilder, ruleInfoList, providerInfoList, aspectInfoList, macroInfoList,
        repositoryRuleInfoList, moduleExtensionInfoList, NATIVE_RULES);
    addMorePredeclared(initialEnvBuilder);

    ImmutableMap<String, Object> initialEnv = initialEnvBuilder.build();

    Map<String, Object> predeclaredSymbols = new HashMap<>();
    predeclaredSymbols.putAll(initialEnv);

    // Add Label constructor if not already present
    addLabelConstructor(predeclaredSymbols);

    Resolver.Module predeclaredResolver = new Resolver.Module() {
      @Override
      public Scope resolve(String name) {
        if (predeclaredSymbols.containsKey(name)) {
          return Scope.PREDECLARED;
        }
        if (!Starlark.UNIVERSE.containsKey(name)) {
          predeclaredSymbols.put(name, FakeDeepStructure.create(name));
          return Scope.PREDECLARED;
        }
        return Resolver.Scope.UNIVERSAL;
      }

      @Override
      public StarlarkType resolveType(String name) throws Resolver.Module.Undefined {
        // For now, throw Undefined - type resolution is not needed for this use case
        throw new Resolver.Module.Undefined("Type resolution not supported");
      }
    };

    // parse & compile (and get doc)
    Program prog;
    StarlarkFile file;
    try {
      file = StarlarkFile.parse(input, FileOptions.DEFAULT);
      moduleDocMap.put(label, getModuleDoc(file));
      prog = Program.compileFile(file, predeclaredResolver);
    } catch (SyntaxError.Exception ex) {
      Event.replayEventsOn(eventHandler, ex.errors());
      throw new StarlarkEvaluationException(ex.getMessage());
    }

    // NOTE: a Program is the syntax tree plus identifiers resolved to bindings.
    // Create module with BazelModuleContext so that StarlarkFunctionInfoExtractor
    // can get the label
    BazelModuleContext moduleContext = BazelModuleContext.create(
        BzlLoadValue.keyForBuild(label),
        RepositoryMapping.EMPTY,
        input.getFile(),
        ImmutableList.of(), // loads will be filled in later
        new byte[0], // bzlTransitiveDigest not needed for extraction
        ImmutableMap.of(), // docCommentsMap not needed for extraction
        ImmutableList.of() // unusedDocCommentLines not needed for extraction
    );
    Module module = Module.withPredeclaredAndData(semantics, predeclaredSymbols, moduleContext);

    // process loads
    for (String load : prog.getLoads()) {
      // Parse the load label - absolute labels start with @ or //, others are
      // relative
      Label from;
      try {
        if (load.startsWith("@") || load.startsWith("//")) {
          from = Label.parseCanonical(load);
        } else {
          // Relative load - resolve against current package
          from = Label.parseCanonical("//" + label.getPackageName() + ":" + load);
        }
      } catch (LabelSyntaxException e) {
        throw new LabelSyntaxException(
            String.format(
                "Invalid load statement '%s' in file %s (%s): %s",
                load,
                label,
                input.getFile(),
                e.getMessage()));
      }
      Path path = pathOfLabel(from);
      try {
        ParserInput loadInput = getInputSource(path.toString());
        Module loadedModule = recursiveEval(loadInput, from, ruleInfoList, providerInfoList, aspectInfoList,
            macroInfoList, repositoryRuleInfoList, moduleExtensionInfoList, moduleDocMap);
        imports.put(load, loadedModule);
        moduleGraph.addEdge(module, loadedModule);
      } catch (StarlarkEvaluationException evalEx) {
        // If a transitive load fails with an evaluation error (e.g., missing symbol),
        // create a stub module to allow best-effort extraction of the top-level file
        Label topLevelLabel = pending.isEmpty() ? null : pending.iterator().next();
        boolean isTransitiveLoad = topLevelLabel != null && !label.equals(topLevelLabel);

        if (isTransitiveLoad) {
          logger.atWarning().log("Transitive load failed for '%s' from %s (best-effort): %s",
              load, label, evalEx.getMessage());

          // Extract symbols that were being loaded
          List<String> loadedSymbols = new ArrayList<>();
          for (Statement stmt : file.getStatements()) {
            if (stmt instanceof LoadStatement) {
              LoadStatement loadStmt = (LoadStatement) stmt;
              if (loadStmt.getImport().getValue().equals(load)) {
                for (LoadStatement.Binding binding : loadStmt.getBindings()) {
                  loadedSymbols.add(binding.getOriginalName().getName());
                }
              }
            }
          }

          // Create stub module with requested symbols
          Module stubModule = createStubModule(load, loadedSymbols);
          logger.atFine().log("Created stub module for failed load '%s' with %d symbols: %s",
              load, stubModule.getGlobals().size(), stubModule.getGlobals().keySet());

          imports.put(load, stubModule);
          moduleGraph.addEdge(module, stubModule);
        } else {
          // For top-level file, propagate the error
          throw evalEx;
        }
      } catch (NoSuchFileException noSuchFileException) {
        // Build load chain message for logging
        StringBuilder loadChain = new StringBuilder();
        loadChain.append("\nLoad chain:\n");
        int depth = 0;
        for (Label l : pending) {
          loadChain.append("  ".repeat(depth)).append(l).append("\n");
          depth++;
        }
        loadChain.append("  ".repeat(depth)).append("-> ").append(load).append(" (NOT FOUND)");

        logger.atFine().log("Failed to load '%s' from %s. Using stub module.%s", load, path, loadChain);

        // Extract symbols that are being loaded from this module
        // Note: we need the original names from the module, not the local aliases
        List<String> loadedSymbols = new ArrayList<>();
        for (Statement stmt : file.getStatements()) {
          if (stmt instanceof LoadStatement) {
            LoadStatement loadStmt = (LoadStatement) stmt;
            if (loadStmt.getImport().getValue().equals(load)) {
              for (LoadStatement.Binding binding : loadStmt.getBindings()) {
                // Use the original name from the loaded module, not the local alias
                loadedSymbols.add(binding.getOriginalName().getName());
              }
            }
          }
        }

        // Create a stub module that returns FakeDeepStructure for requested symbols
        Module stubModule = createStubModule(load, loadedSymbols);

        logger.atFine().log("Created stub module for '%s' with %d symbols: %s", load,
            stubModule.getGlobals().size(), stubModule.getGlobals().keySet());

        imports.put(load, stubModule);
        moduleGraph.addEdge(module, stubModule);
      }
    }

    // Add stubs for any known missing symbols from previous attempts
    LinkedHashSet<String> missingSymbols = missingSymbolsByLabel.getOrDefault(label, new LinkedHashSet<>());
    for (String symbol : missingSymbols) {
      if (!predeclaredSymbols.containsKey(symbol)) {
        predeclaredSymbols.put(symbol, FakeDeepStructure.create(symbol));
        logger.atFine().log("Adding stub for known missing symbol '%s' in %s", symbol, label);
      }
    }

    // Execute with retry mechanism for missing symbols
    // Track the size of missing symbols to detect if we're making progress
    int previousMissingCount = missingSymbols.size();
    int retryCount = 0;
    EvalException lastException = null;

    while (retryCount <= MAX_RETRIES_PER_FILE) {
      try (Mutability mu = Mutability.create("Constellate")) {
        StarlarkThread thread = StarlarkThread.create(mu, semantics, "constellate", SymbolGenerator.createTransient());
        // We use the default print handler, which writes to stderr.
        thread.setLoader(imports::get);
        // Fake Bazel's "export" hack, by which provider symbols
        // bound to global variables take on the name of the global variable.
        thread.setPostAssignHook((name, location, value) -> {
          // Post assign hook now receives: String name, Location location, Object value
          // Handle tuples from provider(init=...) which returns (provider,
          // raw_constructor)
          Object actualValue = value;
          if (value instanceof Tuple && ((Tuple) value).size() == 2
              && ((Tuple) value).get(0) instanceof FakeProviderApi) {
            actualValue = ((Tuple) value).get(0);
          }

          if (actualValue instanceof FakeProviderApi) {
            ((FakeProviderApi) actualValue).setName(name);
          } else if (actualValue instanceof FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) {
            FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier functionIdentifier = (FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) actualValue;
            functionIdentifier.setAssignedName(name);
          }
        });
        Starlark.execFileProgram(prog, module, thread);
        // Success! Break out of retry loop
        break;
      } catch (EvalException ex) {
        lastException = ex;
        String errorMsg = ex.getMessage();

        // Check if this is a "does not contain symbol" error
        MissingSymbolInfo missingInfo = extractMissingSymbol(errorMsg);
        if (missingInfo != null && retryCount < MAX_RETRIES_PER_FILE) {
          // Track this missing symbol
          String trackingKey = missingInfo.filename + ":" + missingInfo.symbol;
          missingSymbols = missingSymbolsByLabel.computeIfAbsent(label, k -> new LinkedHashSet<>());
          boolean isNew = missingSymbols.add(trackingKey);

          if (isNew) {
            logger.atWarning().log(
                "Missing symbol '%s' from file '%s' in %s (retry %d/%d, %d total missing), adding stub and retrying",
                missingInfo.symbol, missingInfo.filename, label, retryCount + 1, MAX_RETRIES_PER_FILE,
                missingSymbols.size());

            // Check if we're making progress (discovering new missing symbols)
            int currentMissingCount = missingSymbols.size();
            if (currentMissingCount <= previousMissingCount) {
              // No progress - we found the same or fewer missing symbols, break circuit
              logger.atWarning().log("No progress made in %s (missing symbols: %d -> %d), stopping retry",
                  label, previousMissingCount, currentMissingCount);
              break;
            }
            previousMissingCount = currentMissingCount;
            retryCount++;

            // Get or create the module for the file that's missing the symbol
            Module targetModule = imports.get(missingInfo.filename);
            if (targetModule == null) {
              // Create a stub module for this file
              logger.atFine().log("Creating stub module for '%s' to add symbol '%s'",
                  missingInfo.filename, missingInfo.symbol);
              targetModule = createStubModule(missingInfo.filename, Arrays.asList(missingInfo.symbol));
              imports.put(missingInfo.filename, targetModule);
            } else {
              // Augment existing module with the missing symbol
              logger.atFine().log("Augmenting module '%s' with stub symbol '%s'",
                  missingInfo.filename, missingInfo.symbol);
              try {
                targetModule.setGlobal(missingInfo.symbol, FakeDeepStructure.create(missingInfo.symbol));
              } catch (Exception e) {
                logger.atWarning().log("Failed to add symbol '%s' to module '%s': %s",
                    missingInfo.symbol, missingInfo.filename, e.getMessage());
              }
            }

            // Recreate the current module for retry
            moduleContext = BazelModuleContext.create(
                BzlLoadValue.keyForBuild(label),
                RepositoryMapping.EMPTY,
                input.getFile(),
                ImmutableList.of(), // loads will be filled in later
                new byte[0], // bzlTransitiveDigest not needed for extraction
                ImmutableMap.of(), // docCommentsMap not needed for extraction
                ImmutableList.of() // unusedDocCommentLines not needed for extraction
            );
            module = Module.withPredeclaredAndData(semantics, predeclaredSymbols, moduleContext);
            continue; // Retry
          } else {
            // Symbol was already added but still failing - no progress
            logger.atWarning().log("Symbol '%s' from '%s' already stubbed but still missing in %s, stopping retry",
                missingInfo.symbol, missingInfo.filename, label);
            break; // Fall through to normal error handling
          }
        } else {
          // Not a missing symbol error, or max retries reached - fall through to normal
          // error handling
          if (retryCount >= MAX_RETRIES_PER_FILE) {
            logger.atWarning().log("Max retries (%d) reached for %s, giving up", MAX_RETRIES_PER_FILE, label);
          }
          break;
        }
      }
    }

    // If we exited the loop with an exception, handle it
    if (lastException != null) {
      EvalException ex = lastException;
      // Handle various evaluation errors gracefully by checking the error message
      String errorMsg = ex.getMessage();
      boolean shouldIgnore = false;
      String ignoreReason = null;

      // Handle deprecated/unknown parameters
      if (errorMsg != null && errorMsg.contains("got unexpected keyword argument")) {
        // List of known deprecated parameters that can be safely ignored
        String[] deprecatedParams = {
            "incompatible_use_toolchain_transition",
            // Add other deprecated parameters here as needed
        };

        for (String param : deprecatedParams) {
          if (errorMsg.contains("'" + param + "'") || errorMsg.contains("\"" + param + "\"")) {
            shouldIgnore = true;
            ignoreReason = "deprecated parameter: " + errorMsg;
            break;
          }
        }
      }

      // Handle type errors for implementation parameter when using fake objects
      if (!shouldIgnore && errorMsg != null && errorMsg.contains("parameter 'implementation'")
          && errorMsg.contains("got value of type") && errorMsg.contains("want 'function'")) {
        shouldIgnore = true;
        ignoreReason = "fake implementation object used where function expected: " + errorMsg;
      }

      // Handle missing/renamed symbols in loaded files
      if (!shouldIgnore && errorMsg != null && errorMsg.contains("does not contain symbol")) {
        // Get the top-level label (first element in pending set)
        Label topLevelLabel = pending.isEmpty() ? null : pending.iterator().next();

        // For transitive loads (not the top-level file), ignore all "does not contain
        // symbol" errors
        // to allow best-effort extraction of the top-level file
        if (topLevelLabel != null && !label.equals(topLevelLabel)) {
          shouldIgnore = true;
          ignoreReason = "missing symbol in transitive load (best-effort extraction): " + errorMsg;
        } else {
          // For the top-level file, only ignore known deprecated symbols
          String[] deprecatedSymbols = {
              "use_cc_toolchain", // Renamed/removed in rules_cc
              // Add other deprecated symbols here as needed
          };

          for (String symbol : deprecatedSymbols) {
            if (errorMsg.contains("'" + symbol + "'") || errorMsg.contains("\"" + symbol + "\"")) {
              shouldIgnore = true;
              ignoreReason = "missing/renamed symbol: " + errorMsg;
              break;
            }
          }
        }
      }

      if (shouldIgnore) {
        logger.atWarning().log("Ignoring error in %s: %s", label, ignoreReason);
        // Keep the partially-evaluated module - it may have some successful definitions
        // The module object already exists and may contain globals that were set before
        // the error
        logger.atFine().log("Continuing with partial evaluation of %s (has %d globals)",
            label, module.getGlobals().size());
      } else {
        throw new StarlarkEvaluationException(ex.getMessageWithStack());
      }
    }

    pending.remove(label);
    loaded.put(label, module);

    // Best-effort enhancement: extract OriginKey and other metadata from real
    // evaluated objects
    try {
      RealObjectEnhancer enhancer = new RealObjectEnhancer(label);
      enhancer.enhance(
          module,
          ruleInfoList,
          providerInfoList,
          aspectInfoList,
          macroInfoList,
          repositoryRuleInfoList,
          moduleExtensionInfoList);
      logger.atFine().log("Successfully enhanced module %s with real object metadata", label);
    } catch (Exception e) {
      // Enhancement is best-effort - log but don't fail evaluation
      logger.atWarning().withCause(e).log(
          "Failed to enhance module %s with real object metadata", label);
    }

    return module;
  }

  public Path pathOfLabel(Label label) throws EvalException {
    // workspaceRoot is either the empty string for labels like '//pkg:target'
    // or 'external/repo' for labels like `@repo//pkg:target`.
    String workspaceRoot = label.getWorkspaceRootForStarlarkOnly(semantics);
    if (workspaceRoot.isEmpty()) {
      // Local workspace file
      logger.atFine().log("Resolving local workspace file: %s", label);
      return Paths.get(label.toPathFragment().toString());
    }
    // External workspace file
    logger.atFine().log("Resolving external workspace file: %s (workspace_root=%s)", label, workspaceRoot);
    return Paths.get(workspaceRoot, label.toPathFragment().toString());
  }

  /**
   * Holds information about a missing symbol from a load statement error.
   */
  private static class MissingSymbolInfo {
    final String filename;
    final String symbol;

    MissingSymbolInfo(String filename, String symbol) {
      this.filename = filename;
      this.symbol = symbol;
    }
  }

  /**
   * Extracts the missing symbol info from a "does not contain symbol" error
   * message.
   * Error format: "file 'X' does not contain symbol 'Y'"
   * Returns null if the message doesn't match this format.
   */
  private static MissingSymbolInfo extractMissingSymbol(String errorMsg) {
    if (errorMsg == null || !errorMsg.contains("does not contain symbol")) {
      return null;
    }

    // Extract filename: file 'filename'
    int fileStart = errorMsg.indexOf("file '");
    if (fileStart == -1) {
      fileStart = errorMsg.indexOf("file \"");
    }
    if (fileStart == -1) {
      return null;
    }
    fileStart += 6; // length of "file '"
    int fileEnd = errorMsg.indexOf("'", fileStart);
    if (fileEnd == -1) {
      fileEnd = errorMsg.indexOf("\"", fileStart);
    }
    if (fileEnd == -1 || fileEnd <= fileStart) {
      return null;
    }
    String filename = errorMsg.substring(fileStart, fileEnd);

    // Extract symbol: symbol 'symbolName'
    int symbolStart = errorMsg.lastIndexOf("symbol '");
    if (symbolStart == -1) {
      symbolStart = errorMsg.lastIndexOf("symbol \"");
    }
    if (symbolStart == -1) {
      return null;
    }
    symbolStart += 8; // length of "symbol '"
    int symbolEnd = errorMsg.indexOf("'", symbolStart);
    if (symbolEnd == -1) {
      symbolEnd = errorMsg.indexOf("\"", symbolStart);
    }
    if (symbolEnd == -1 || symbolEnd <= symbolStart) {
      return null;
    }
    String symbol = errorMsg.substring(symbolStart, symbolEnd);

    return new MissingSymbolInfo(filename, symbol);
  }

  /**
   * Converts a Starlark value to a Value proto message.
   * Supports string, int, bool, and list types.
   * Returns null for unsupported types.
   */
  private StarlarkProtos.Value convertToValue(Object value, String name) {
    if (value instanceof String) {
      return StarlarkProtos.Value.newBuilder().setString((String) value).build();
    } else if (value instanceof StarlarkInt) {
      StarlarkInt si = (StarlarkInt) value;
      try {
        long longValue = si.toLong("global constant");
        return StarlarkProtos.Value.newBuilder().setInt(longValue).build();
      } catch (EvalException e) {
        logger.atWarning().log("Could not convert StarlarkInt to long for %s: %s", name, e.getMessage());
        return null;
      }
    } else if (value instanceof Boolean) {
      return StarlarkProtos.Value.newBuilder().setBool((Boolean) value).build();
    } else if (value instanceof net.starlark.java.eval.StarlarkList) {
      net.starlark.java.eval.StarlarkList<?> list = (net.starlark.java.eval.StarlarkList<?>) value;
      StarlarkProtos.ValueList.Builder listBuilder = StarlarkProtos.ValueList.newBuilder();

      for (Object item : list) {
        StarlarkProtos.Value itemValue = convertToValue(item, name + "[]");
        if (itemValue != null) {
          listBuilder.addValue(itemValue);
        } else {
          // If we can't convert an item, log and skip it
          logger.atFine().log("Skipping unsupported list item type in %s: %s", name,
              item == null ? "null" : item.getClass().getName());
        }
      }

      return StarlarkProtos.Value.newBuilder().setList(listBuilder.build()).build();
    }

    // Unsupported type
    return null;
  }

  public ParserInput getInputSource(String bzlWorkspacePath) throws IOException {
    logger.atFine().log("Searching for input source: %s (roots: %s)", bzlWorkspacePath, depRoots);
    for (String rootPath : depRoots) {
      String filepath = rootPath + "/" + bzlWorkspacePath;
      if (fileAccessor.fileExists(filepath)) {
        logger.atFine().log("Found input source: %s at %s", bzlWorkspacePath, filepath);
        return fileAccessor.inputSource(filepath);
      } else {
        logger.atFine().log("Input source not found at: %s", filepath);
      }
    }

    logger.atFine().log("Failed to resolve input source: %s (searched roots: %s)", bzlWorkspacePath, depRoots);

    // All depRoots attempted and no valid file was found.
    throw new NoSuchFileException(bzlWorkspacePath);
  }

  /**
   * Creates a stub module that returns FakeDeepStructure for requested symbols.
   * This allows evaluation to continue even when a load fails.
   */
  private Module createStubModule(String loadLabel, List<String> symbols) {
    Map<String, Object> predeclared = new HashMap<>();

    // Populate predeclared with FakeDeepStructure for each requested symbol
    for (String symbol : symbols) {
      predeclared.put(symbol, FakeDeepStructure.create(symbol));
      logger.atFine().log("Created stub symbol '%s' in failed load: %s", symbol, loadLabel);
    }

    // Create module with predeclared bindings
    Module stubModule = Module.withPredeclared(semantics, predeclared);

    // Set each symbol as a global in the module
    // This is necessary because Module.withPredeclared only makes symbols available
    // during
    // execution, but load statements look at module.getGlobals()
    for (String symbol : symbols) {
      stubModule.setGlobal(symbol, FakeDeepStructure.create(symbol));
      logger.atFine().log("Created stub symbol '%s' in failed load: %s", symbol, loadLabel);
    }

    return stubModule;
  }

  /**
   * Load native rule documentation from bundled binary proto resources.
   * These protos are generated from Build Encyclopedia entry points and bundled
   * into the JAR during build.
   *
   * Called once at class initialization time and cached in NATIVE_RULES static
   * field.
   * Returns an immutable map for thread safety in worker processes.
   *
   * @return Immutable map of rule name to RuleInfo for all native Bazel rules
   */
  private static Map<String, RuleInfo> loadNativeRulesFromResources() {
    Map<String, RuleInfo> nativeRules = new HashMap<>();

    // List of resource paths for native rule protos (bundled in JAR)
    String[] resourcePaths = {
        "/main/starlark/docgen/gen_be_java_stardoc_proto.binaryproto",
        "/main/starlark/docgen/gen_be_cpp_stardoc_proto.binaryproto",
        "/main/starlark/docgen/gen_be_python_stardoc_proto.binaryproto",
        "/main/starlark/docgen/gen_be_proto_stardoc_proto.binaryproto",
        "/main/starlark/docgen/gen_be_shell_stardoc_proto.binaryproto",
        "/main/starlark/docgen/gen_be_objc_stardoc_proto.binaryproto",
    };

    for (String resourcePath : resourcePaths) {
      try (java.io.InputStream stream = StarlarkEvaluator.class.getResourceAsStream(resourcePath)) {
        if (stream == null) {
          logger.atFine().log("Native rule proto resource not found: %s", resourcePath);
          continue;
        }

        // Parse binary proto from resource
        com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo moduleInfo = com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo
            .parseFrom(
                stream,
                com.google.protobuf.ExtensionRegistry.getEmptyRegistry());

        // Extract RuleInfo for each rule
        for (RuleInfo ruleInfo : moduleInfo.getRuleInfoList()) {
          String ruleName = ruleInfo.getRuleName();
          nativeRules.put(ruleName, ruleInfo);
          // Also add mapping from short name (e.g., "java_library") to RuleInfo
          // The ruleName might be like "library_rules.java_library"
          if (ruleName.contains(".")) {
            String shortName = ruleName.substring(ruleName.lastIndexOf('.') + 1);
            nativeRules.put(shortName, ruleInfo);
            logger.atFine().log("Loaded native rule from resource %s: %s (short: %s)", resourcePath, ruleName,
                shortName);
          } else {
            logger.atFine().log("Loaded native rule from resource %s: %s", resourcePath, ruleName);
          }
        }
      } catch (java.io.IOException e) {
        logger.atWarning().log("Failed to load native rules from resource %s: %s",
            resourcePath, e.getMessage());
      }
    }

    logger.atInfo().log("Loaded %d native rules from bundled resources", nativeRules.size());

    // Return immutable map for thread safety in worker processes
    return ImmutableMap.copyOf(nativeRules);
  }

  /**
   * Adds a Label constructor to the predeclared symbols, replacing any existing
   * stub.
   * This allows Starlark code to call Label("foo.bzl") with package-relative
   * labels.
   */
  private static void addLabelConstructor(Map<String, Object> predeclaredSymbols) {
    // Replace any existing Label (e.g., FakeDeepStructure) with our proper
    // implementation
    predeclaredSymbols.put("Label", new StarlarkCallable() {
      @Override
      public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) throws EvalException {
        if (args.size() != 1) {
          throw Starlark.errorf("Label() takes exactly 1 argument (%d given)", args.size());
        }

        String labelStr = Starlark.str(args.get(0), thread.getSemantics());

        // Get the current module's label from the BazelModuleContext
        BazelModuleContext moduleContext;
        try {
          moduleContext = BazelModuleContext.ofInnermostBzlOrFail(thread, "Label");
        } catch (EvalException e) {
          // Fallback: try to parse as canonical label if we can't get module context
          try {
            return Label.parseCanonical(labelStr);
          } catch (LabelSyntaxException ex) {
            throw Starlark.errorf("Label: %s", ex.getMessage());
          }
        }

        Label currentLabel = moduleContext.label();
        try {
          // Handle absolute labels (starting with // or @)
          if (labelStr.startsWith("//") || labelStr.startsWith("@")) {
            return Label.parseCanonical(labelStr);
          }

          // Handle relative labels - resolve against current package
          // Relative labels like "foo.bzl" or ":foo.bzl" are in the current package
          String targetName = labelStr.startsWith(":") ? labelStr.substring(1) : labelStr;
          return Label.create(currentLabel.getPackageIdentifier(), targetName);
        } catch (LabelSyntaxException e) {
          throw Starlark.errorf("Label: %s", e.getMessage());
        }
      }

      @Override
      public String getName() {
        return "Label";
      }
    });
  }

  private static void addMorePredeclared(ImmutableMap.Builder<String, Object> env) {
    // Add dummy declarations that would come from packages.StarlarkLibrary.COMMON
    // were Constellate allowed to depend on it. See hack for select below.
    env.put("json", Json.INSTANCE);
    env.put("proto", new ProtoModule());
    env.put("depset", new StarlarkCallable() {
      @Override
      public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) {
        // Accept any arguments, return empty Depset.
        return Depset.of(Object.class, NestedSetBuilder.emptySet(Order.STABLE_ORDER));
      }

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
    env.put("select", new StarlarkCallable() {
      @Override
      public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) throws EvalException {
        // Accept dict as first positional argument, return first value
        if (args.size() > 0) {
          for (Map.Entry<?, ?> e : ((Dict<?, ?>) args.get(0)).entrySet()) {
            return e.getValue();
          }
        }
        throw Starlark.errorf("select: empty dict");
      }

      @Override
      public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) throws EvalException {
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

    // Override fail() to log instead of throwing an exception
    env.put("fail", new StarlarkCallable() {
      @Override
      public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) throws EvalException {
        String message = args.size() > 0 ? Starlark.str(args.get(0), thread.getSemantics()) : "fail() called";
        logger.atFine().log("Starlark fail() called at %s: %s", thread.getCallerLocation(), message);
        return Starlark.NONE;
      }

      @Override
      public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) throws EvalException {
        String message = positional.length > 0 ? Starlark.str(positional[0], thread.getSemantics()) : "fail() called";
        logger.atFine().log("Starlark fail() called at %s: %s", thread.getCallerLocation(), message);
        return Starlark.NONE;
      }

      @Override
      public String getName() {
        return "fail";
      }
    });
  }

  @StarlarkBuiltin(name = "ProtoModule", doc = "")
  private static final class ProtoModule implements StarlarkValue {
    @StarlarkMethod(name = "encode_text", doc = ".", parameters = { @Param(name = "x") })
    public String encodeText(Object x) {
      return "";
    }
  }

  /**
   * Exception thrown when Starlark evaluation fails (due to malformed Starlark).
   */
  @VisibleForTesting
  static class StarlarkEvaluationException extends Exception {
    public StarlarkEvaluationException(String message) {
      super(message);
    }

    public StarlarkEvaluationException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  static class ModuleEvalContext {
    final List<RuleInfoWrapper> ruleInfoList = new ArrayList<>();
    final List<ProviderInfoWrapper> providerInfoList = new ArrayList<>();
    final List<AspectInfoWrapper> aspectInfoList = new ArrayList<>();

    ModuleEvalContext() {
    }
  }

  private static Object resolveFunctionIdentifier(StarlarkFunction fn, Identifier id)
      throws EvalException, InterruptedException {
    Resolver.Binding bind = id.getBinding();
    switch (bind.getScope()) {
      case LOCAL:
        return null;
      case CELL:
        return null;
      case FREE:
        return null;
      case GLOBAL:
        // getGlobal() is now private, use the module's globals map instead
        return fn.getModule().getGlobals().get(id.getName());
      case PREDECLARED:
        return fn.getModule().getPredeclared(id.getName());
      case UNIVERSAL:
        return Starlark.UNIVERSE.get(id.getName());
      default:
        throw new IllegalStateException(bind.toString());
    }
  }

  /**
   * Analyzes a function's body to find calls to rules, aspects, and macros.
   * Returns a list of names of rules/aspects/macros that this function calls.
   * This helps identify "wrapper functions" (traditional Bazel macros).
   */
  private static List<String> findRuleAndMacroCallsInFunction(
      StarlarkFunction fn,
      Map<String, RuleInfo> ruleInfoMap,
      Map<String, AspectInfo> aspectInfoMap,
      Map<String, MacroInfo> macroInfoMap,
      List<RuleInfoWrapper> ruleInfoList,
      Map<String, RuleInfo> nativeRules) {
    List<String> calledNames = new ArrayList<>();

    try {
      // Get the function body statements via the public accessor
      net.starlark.java.syntax.Resolver.Function resolverFn = fn.getResolverFunction();
      if (resolverFn == null) {
        // Some functions (e.g., built-in or wrapped functions) may not have a resolver
        // function
        logger.atFine().log("Function %s has no resolver function (likely built-in or wrapped)", fn.getName());
        return calledNames;
      }

      ImmutableList<net.starlark.java.syntax.Statement> body = resolverFn.getBody();

      // Walk through each statement looking for call expressions
      for (net.starlark.java.syntax.Statement stmt : body) {
        findCallsInStatement(stmt, fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules,
            calledNames);
      }
    } catch (Exception e) {
      // If we can't analyze the function, just return empty list
      String errorMsg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
      logger.atWarning().log("Could not analyze function %s for rule/macro calls: %s",
          fn.getName(), errorMsg);
    }

    return calledNames;
  }

  /**
   * Recursively searches a statement for calls to rules, aspects, and macros.
   */
  private static void findCallsInStatement(
      net.starlark.java.syntax.Statement stmt,
      StarlarkFunction fn,
      Map<String, RuleInfo> ruleInfoMap,
      Map<String, AspectInfo> aspectInfoMap,
      Map<String, MacroInfo> macroInfoMap,
      List<RuleInfoWrapper> ruleInfoList,
      Map<String, RuleInfo> nativeRules,
      List<String> calledNames) {

    if (stmt instanceof net.starlark.java.syntax.ExpressionStatement) {
      Expression expr = ((net.starlark.java.syntax.ExpressionStatement) stmt).getExpression();
      findCallsInExpression(expr, fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules, calledNames);
    } else if (stmt instanceof net.starlark.java.syntax.AssignmentStatement) {
      Expression rhs = ((net.starlark.java.syntax.AssignmentStatement) stmt).getRHS();
      findCallsInExpression(rhs, fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules, calledNames);
    } else if (stmt instanceof net.starlark.java.syntax.IfStatement) {
      net.starlark.java.syntax.IfStatement ifStmt = (net.starlark.java.syntax.IfStatement) stmt;
      for (net.starlark.java.syntax.Statement s : ifStmt.getThenBlock()) {
        findCallsInStatement(s, fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules, calledNames);
      }
      // getElseBlock() returns null if there's no else clause
      if (ifStmt.getElseBlock() != null) {
        for (net.starlark.java.syntax.Statement s : ifStmt.getElseBlock()) {
          findCallsInStatement(s, fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules, calledNames);
        }
      }
    } else if (stmt instanceof net.starlark.java.syntax.ForStatement) {
      net.starlark.java.syntax.ForStatement forStmt = (net.starlark.java.syntax.ForStatement) stmt;
      for (net.starlark.java.syntax.Statement s : forStmt.getBody()) {
        findCallsInStatement(s, fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules, calledNames);
      }
    }
    // Other statement types (def, return, etc.) - we could extend this as needed
  }

  /**
   * Recursively searches an expression for calls to rules, aspects, and macros.
   */
  private static void findCallsInExpression(
      Expression expr,
      StarlarkFunction fn,
      Map<String, RuleInfo> ruleInfoMap,
      Map<String, AspectInfo> aspectInfoMap,
      Map<String, MacroInfo> macroInfoMap,
      List<RuleInfoWrapper> ruleInfoList,
      Map<String, RuleInfo> nativeRules,
      List<String> calledNames) {

    if (expr instanceof CallExpression) {
      CallExpression call = (CallExpression) expr;
      Expression callTarget = call.getFunction();

      // Check if this is a simple identifier call (e.g., my_rule(...))
      if (callTarget instanceof Identifier) {
        String name = ((Identifier) callTarget).getName();

        // Check if this name is a rule, aspect, or macro
        boolean isRuleOrMacro = ruleInfoMap.containsKey(name) || aspectInfoMap.containsKey(name)
            || macroInfoMap.containsKey(name);

        // Also check ruleInfoList for rules from loaded modules that might not be in
        // the map
        if (!isRuleOrMacro) {
          for (RuleInfoWrapper wrapper : ruleInfoList) {
            if (wrapper.getIdentifierFunction() instanceof PostAssignHookAssignableIdentifier) {
              PostAssignHookAssignableIdentifier ident = (PostAssignHookAssignableIdentifier) wrapper
                  .getIdentifierFunction();
              if (ident.getAssignedName().equals(name)) {
                isRuleOrMacro = true;
                break;
              }
            }
          }
        }

        if (isRuleOrMacro && !calledNames.contains(name)) {
          calledNames.add(name);
        }
      }
      // Check if this is a DotExpression call (e.g., native.genrule(...))
      else if (callTarget instanceof DotExpression) {
        DotExpression dotExpr = (DotExpression) callTarget;
        Expression object = dotExpr.getObject();
        String field = dotExpr.getField().getName();

        // Check if this is a call to native.*
        if (object instanceof Identifier && ((Identifier) object).getName().equals("native")) {
          // Check if this is a known native rule
          if (NATIVE_RULES.containsKey(field)) {
            String nativeCallName = "native." + field;
            if (!calledNames.contains(nativeCallName)) {
              calledNames.add(nativeCallName);
            }
          }
        }
      }

      // Also check arguments for nested calls
      for (Argument arg : call.getArguments()) {
        if (arg instanceof Argument.Positional) {
          findCallsInExpression(((Argument.Positional) arg).getValue(), fn, ruleInfoMap, aspectInfoMap, macroInfoMap,
              ruleInfoList, nativeRules, calledNames);
        } else if (arg instanceof Argument.Keyword) {
          findCallsInExpression(((Argument.Keyword) arg).getValue(), fn, ruleInfoMap, aspectInfoMap, macroInfoMap,
              ruleInfoList, nativeRules, calledNames);
        }
      }
    } else if (expr instanceof net.starlark.java.syntax.ListExpression) {
      for (Expression elem : ((net.starlark.java.syntax.ListExpression) expr).getElements()) {
        findCallsInExpression(elem, fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules,
            calledNames);
      }
    } else if (expr instanceof net.starlark.java.syntax.DictExpression) {
      for (net.starlark.java.syntax.DictExpression.Entry entry : ((net.starlark.java.syntax.DictExpression) expr)
          .getEntries()) {
        findCallsInExpression(entry.getKey(), fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules,
            calledNames);
        findCallsInExpression(entry.getValue(), fn, ruleInfoMap, aspectInfoMap, macroInfoMap, ruleInfoList, nativeRules,
            calledNames);
      }
    }
    // We could extend this to handle other expression types as needed
  }
}
