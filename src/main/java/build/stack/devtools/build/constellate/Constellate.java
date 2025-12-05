package build.stack.devtools.build.constellate;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
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

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
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
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.lib.json.Json;
import net.starlark.java.syntax.Argument;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.CallExpression;
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
public class Constellate {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  // eventHandler is used to replay events when we get a compile error.
  private final EventHandler eventHandler = new SystemOutEventHandler();
  // fileAccessor helps load files.
  private final StarlarkFileAccessor fileAccessor;
  // depRoots is the list of module root dirs.
  private final List<String> depRoots;
  // workspaceName is the name if the external workspace, if defined. It
  // inflences the 'pathOfLabel' function.
  private final String workspaceName;
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

  public Constellate(StarlarkSemantics semantics, StarlarkFileAccessor fileAccessor, String workspaceName,
      List<String> depRoots) {
    this.semantics = semantics;
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

    Module module = recursiveEval(input, label, ruleInfoList, providerInfoList, aspectInfoList, macroInfoList, repositoryRuleInfoList, moduleExtensionInfoList, moduleDocMap);

    logger.atFine().log("\n\nresolving module globals: %s", label);

    resolveGlobals(
        module,
        ruleInfoMap,
        providerInfoMap,
        userDefinedFunctionMap,
        aspectInfoMap,
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
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      ImmutableMap.Builder<String, ProviderInfo> providerInfoMap,
      ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctionMap,
      ImmutableMap.Builder<String, AspectInfo> aspectInfoMap,
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

    // logger.atInfo().log("module globals: %s", sortedBindings);

    // calledWithKwargs represents a function identifier that was called using
    // kwargs from a user defined function. For example, if the body of `def
    // _buildifier(**kwargs)` calls `buildifier(**kwargs)`, the we store a
    // mapping [buildifier<String>, _buildifier<string>]. There can be multiple
    // alternate rules called by a macro (e.g. go_transition_wrapper=go_test and
    // go_transition_wrapper=go_binary)
    ImmutableListMultimap.Builder<String, Collection<String>> calledWithKwargs = ImmutableListMultimap.builder();

    // Log all exported symbols
    for (Entry<String, Module> loadedModule : imports.entrySet()) {
      TreeMap<String, Object> exports = new TreeMap<>(loadedModule.getValue().getGlobals());
      for (Entry<String, Object> export : exports.entrySet()) {
        logger.atFine().log("%s top-level symbol %s -> %s",
            loadedModule.getKey(), export.getKey(),
            export.getValue().getClass().getName());
      }
    }

    // logger.atFine().log("top-level global objects of %s: %s", label,
    // sortedBindings.size());
    logger.atFine().log("resolving module %s: rules: %s, providers: %s, aspects: %s, macros: %s, repository_rules: %s, module_extensions: %s",
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
        RuleInfo ruleInfo = wrapper.getRuleInfo().build();
        // Use symbol name as the rule name only if not already set in the call to
        // rule().
        if ("".equals(ruleInfo.getRuleName())) {
          // We make a copy so that additional exports are not affected by setting the
          // rule name on
          // this builder
          ruleInfo = ruleInfo.toBuilder().setRuleName(envEntry.getKey()).build();
        }
        Location loc = wrapper.getLocation();
        logger.atFine().log("global rule %s", ruleInfo.getRuleName());
        ruleInfoMap.put(ruleInfo.getRuleName(), ruleInfo);
        StarlarkProtos.SymbolLocation symbolLocation = StarlarkProtos.SymbolLocation.newBuilder()
            .setName(ruleInfo.getRuleName())
            .setStart(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .setEnd(StarlarkProtos.Position.newBuilder().setLine(loc.line()).setCharacter(loc.column()).build())
            .build();
        starlarkModule.addSymbolLocation(symbolLocation);
      }

      // +++ PROVIDERS
      if (providerInfos.containsKey(envEntry.getValue())) {
        ProviderInfo.Builder providerInfoBuild = providerInfos.get(envEntry.getValue()).getProviderInfo();
        ProviderInfo providerInfo = providerInfoBuild.setProviderName(envEntry.getKey()).build();
        logger.atFine().log("global provider %s", envEntry.getKey());
        providerInfoMap.put(envEntry.getKey(), providerInfo);
      }

      // +++ FUNCTIONS
      if (envEntry.getValue() instanceof StarlarkFunction) {
        StarlarkFunction userDefinedFunction = (StarlarkFunction) envEntry.getValue();
        logger.atFine().log("global function %s", envEntry.getKey());

        if (userDefinedFunction.hasKwargs()) {
          resolveFunctionKwargs(module, envEntry.getKey(), userDefinedFunction, calledWithKwargs);
        }
        userDefinedFunctionMap.put(envEntry.getKey(), userDefinedFunction);
      }

      // +++ STRUCTS
      if (envEntry.getValue() instanceof FakeStructApi) {
        String namespaceName = envEntry.getKey();
        FakeStructApi namespace = (FakeStructApi) envEntry.getValue();
        logger.atFine().log("global struct %s.%s", namespaceName, namespace);
        putStructFields(namespaceName, namespace, userDefinedFunctionMap);
      } else if (envEntry.getValue() instanceof String) {
        String s = (String) envEntry.getValue();
        starlarkModule.putGlobal(envEntry.getKey(), StarlarkProtos.ValueInfo.newBuilder().setString(s).build());
        // } else if (envEntry.getValue() instanceof String) {
        // String s = (String) envEntry.getValue();
        // starlarkModule.addValue(StarlarkProtos.ValueInfo.newBuilder().setString(s).build());
      }

      // +++ ASPECTS
      if (aspectFunctions.containsKey(envEntry.getValue())) {
        AspectInfo.Builder aspectInfoBuild = aspectFunctions.get(envEntry.getValue()).getAspectInfo();
        AspectInfo aspectInfo = aspectInfoBuild.setAspectName(envEntry.getKey()).build();
        logger.atFine().log("global aspect %s", envEntry.getKey());
        aspectInfoMap.put(envEntry.getKey(), aspectInfo);
      }

      // +++ MACROS
      if (macroFunctions.containsKey(envEntry.getValue())) {
        MacroInfoWrapper wrapper = macroFunctions.get(envEntry.getValue());
        MacroInfo macroInfo = wrapper.getMacroInfo().setMacroName(envEntry.getKey()).build();
        logger.atFine().log("global macro %s", envEntry.getKey());
        // Note: We'll need to add a macroInfoMap parameter in the future when we have
        // a ModuleInfo proto that includes MacroInfo. For now, we just log it.
        // macroInfoMap.put(envEntry.getKey(), macroInfo);
      }

      // +++ REPOSITORY RULES
      if (repositoryRuleFunctions.containsKey(envEntry.getValue())) {
        RepositoryRuleInfoWrapper wrapper = repositoryRuleFunctions.get(envEntry.getValue());
        RepositoryRuleInfo repositoryRuleInfo = wrapper.getRepositoryRuleInfo().setRuleName(envEntry.getKey()).build();
        logger.atFine().log("global repository_rule %s", envEntry.getKey());
        // Note: We'll need to add a repositoryRuleInfoMap parameter in the future when we have
        // a ModuleInfo proto that includes RepositoryRuleInfo. For now, we just log it.
        // repositoryRuleInfoMap.put(envEntry.getKey(), repositoryRuleInfo);
      }

      // +++ MODULE EXTENSIONS
      if (moduleExtensionObjects.containsKey(envEntry.getValue())) {
        ModuleExtensionInfoWrapper wrapper = moduleExtensionObjects.get(envEntry.getValue());
        ModuleExtensionInfo moduleExtensionInfo = wrapper.getModuleExtensionInfo().setExtensionName(envEntry.getKey()).build();
        logger.atFine().log("global module_extension %s", envEntry.getKey());
        // Note: We'll need to add a moduleExtensionInfoMap parameter in the future when we have
        // a ModuleInfo proto that includes ModuleExtensionInfo. For now, we just log it.
        // moduleExtensionInfoMap.put(envEntry.getKey(), moduleExtensionInfo);
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

  }

  private void resolveFunctionMacros(
      ImmutableMap.Builder<String, RuleInfo> ruleInfoMap,
      List<RuleInfoWrapper> ruleInfoList,
      ImmutableListMultimap<String, Collection<String>> rulesCalledWithKwargs,
      ImmutableMap<String, StarlarkFunction> userFunctions) {

    for (String ruleName : rulesCalledWithKwargs.keys()) {
      ImmutableMap<String, RuleInfo> rules = ruleInfoMap.build();
      if (!rules.containsKey(ruleName)) {
        logger.atFine().log("resolveFunctionMacros: skipping rule %s (unknown)", ruleName);
        continue;
      }
      RuleInfo ruleInfo = rules.get(ruleName);
      for (Collection<String> macroNames : rulesCalledWithKwargs.get(ruleName)) {
        for (String macroName : macroNames) {
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
          } catch (DocstringParseException dspex) {
            // best-effort, ignore error
          }

          ruleInfoMap.put(macroName, macroInfo.build());
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
      // String name = ruleWrapper.getIdentifierFunction().getName();
      // logger.atFine().log("checking if rule %s was called by a macro", name);
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
          } catch (DocstringParseException dspex) {
            // best-effort, ignore error
          }

          ruleInfoMap.put(caller, ruleInfo.build());
          logger.atFine().log("global macro %s (rule from %s, called by %s)", caller, name, function.getName());
        }
      }
    }
  }

  static void resolveFunctionKwargs(
      Module module,
      String globalName,
      StarlarkFunction userDefinedFunction,
      ImmutableListMultimap.Builder<String, Collection<String>> calledWithKwargs) {
    logger.atFine().log("** global function %s has kwargs", globalName);

    NodeVisitor checker = new NodeVisitor() {
      Stack<Node> stack = new Stack<>();

      @Override
      public void visit(Node node) {
        stack.push(node);
        super.visit(node);
        stack.pop();
      }

      // Record f(*args) and f(**kwargs) calls.
      void recordStarArgs(CallExpression call) {
        for (Argument arg : call.getArguments()) {
          if (arg instanceof Argument.StarStar) {
            logger.atFine().log("STAR STAR %s: %s", arg.getStartLocation(), arg.getValue());
            for (int i = stack.size() - 1; i >= 0; i--) {
              Node parent = stack.get(i);
              if (parent instanceof CallExpression) {
                CallExpression parentCall = (CallExpression) parent;
                Expression parentCallExpr = parentCall.getFunction();
                if (parentCallExpr instanceof Identifier) {
                  Identifier ident = (Identifier) parentCallExpr;
                  try {
                    logger.atFine().log("  ** call-expression %s gets kwargs from %s (%s)", ident.getName(),
                        globalName, module.resolve(ident.getName()));
                    logger.atFine().log("  ** macro receiver %s binding is %s",
                        ident.getName(), ident.getBinding().getClass().getName());

                    Object resolved = resolveFunctionIdentifier(userDefinedFunction, ident);
                    if (resolved != null) {
                      logger.atFine().log("  ** resolved binding is %s",
                          resolved.getClass().getName());
                    } else {
                      logger.atFine().log("  ** unresolved binding!");
                    }
                    if (resolved instanceof FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) {
                      FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier ruleIdent = (FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) resolved;
                      calledWithKwargs.put(ruleIdent.getAssignedName(), ImmutableList.of(globalName));
                    } else {
                      calledWithKwargs.put(ident.getName(), ImmutableList.of(globalName));
                    }
                  } catch (InterruptedException iEx) {
                    logger.atFine().log("  ** parent-call-expression interrupt exception: %s", iEx);
                  } catch (EvalException evalEx) {
                    logger.atFine().log("  ** parent-call-expression eval exception: %s", evalEx);
                  } catch (Module.Undefined undef) {
                    logger.atFine().log("  ** parent-call-expression is undefined: %s", undef);
                  }
                }
                break;
              }
            }
          } else if (arg instanceof Argument.Star) {
            logger.atFine().log("STAR %s: %s", arg.getStartLocation(), arg.getValue());
          }
        }
      }

      @Override
      public void visit(CallExpression node) {
        recordStarArgs(node);
        super.visit(node);
      }
    };

    // Note: StarlarkFunction no longer exposes getResolverFunction() in modern Bazel.
    // Macro kwargs detection is disabled until an alternative API is found.
    // checker.visitAll(userDefinedFunction.getResolverFunction().getBody());
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

  // /**
  // *
  // */
  // private static RuleInfo getTargetRuleInfo(
  // StarlarkFunction function, // a user defined function
  // Module module, // the module to which the function is a top-level
  // ImmutableListMultimap<Module, Collection<Module>> loadMappings, // the load
  // mappings
  // Digraph<Module> modules, // graph of modules and those they load
  // ImmutableListMultimap<Module,LoadStatement> moduleLoads,
  // ImmutableListMultimap<StarlarkFunction, Collection<CallExpression>>
  // kwargsCallers, // mapping from function to the call expressions given kwargs
  // ) {
  // StarlarkProtos.Module currentModule = module;
  // Stack<StarlarkFunction> current = new Stack();
  // stack.add(current);

  // while (!stack.isEmpty()) {
  // StarlarkFunction currentFunction = stack.pop();

  // // step 1: does this function pass kwargs to a call-expression within the
  // // function body?
  // if (!calledExprs.containsKey(currentFunction)) {
  // continue;
  // }
  // Collection<CallExpression> calledExprs = kwargsCallers.get(currentFunction);

  // // step 3: seek for the loaded symbol that was called by the function.
  // Object targetOfKwargs = null;

  // LOADS:
  // for (Collection<LoadStatement> loads : moduleLoads.get(currentModule)) {
  // for (LoadStatement load : loads) {
  // for (LoadStatement.Binding binding : lo) {
  // String localName = binding.getLocalName();
  // for (CallExpression callExpr : calledExprs()) {
  // if (!(callExpr.getFunction() instanceof Identifier)) {
  // continue;
  // }
  // Identifier ident = (Identifier)callExpr.getFunction();
  // if (!ident.getName().equals(localName)) {
  // Module targetModule =
  // continue;
  // }
  // // we found the thing the current function passed its kwargs to.
  // }
  // }
  // }
  // }
  // Collection<Module> modules = loadMappings.get(current);
  // if (modules == null || modules.isEmpty()) {
  // return null;
  // }
  // for (Module loaded : modules) {
  // // check all exported symbols
  // for (Entry<String, Object> global : new
  // TreeMap(loaded.getGlobal()).entrySet()) {
  // // ignore private symbols
  // if (global.getKey().startsWith("_")) {
  // continue;
  // }
  // }
  // }

  // }
  // }

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
    FakeApi.addPredeclared(initialEnvBuilder, ruleInfoList, providerInfoList, aspectInfoList, macroInfoList, repositoryRuleInfoList, moduleExtensionInfoList);
    addMorePredeclared(initialEnvBuilder);

    ImmutableMap<String, Object> initialEnv = initialEnvBuilder.build();

    Map<String, Object> predeclaredSymbols = new HashMap<>();
    predeclaredSymbols.putAll(initialEnv);

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
    Module module = Module.withPredeclared(semantics, predeclaredSymbols);

    // process loads
    for (String load : prog.getLoads()) {
      // Parse the load label - absolute labels start with @ or //, others are relative
      Label from;
      if (load.startsWith("@") || load.startsWith("//")) {
        from = Label.parseCanonical(load);
      } else {
        // Relative load - resolve against current package
        from = Label.parseCanonical("//" + label.getPackageName() + ":" + load);
      }
      Path path = pathOfLabel(from);
      try {
        ParserInput loadInput = getInputSource(path.toString());
        Module loadedModule = recursiveEval(loadInput, from, ruleInfoList, providerInfoList, aspectInfoList,
            macroInfoList, repositoryRuleInfoList, moduleExtensionInfoList, moduleDocMap);
        imports.put(load, loadedModule);
        moduleGraph.addEdge(module, loadedModule);
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

        logger.atWarning().log("Failed to load '%s' from %s. Using stub module.%s", load, path, loadChain);

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

        // Debug: verify what's in the stub module
        logger.atInfo().log("Stub module globals for '%s': %s", load, stubModule.getGlobals().keySet());

        imports.put(load, stubModule);
        moduleGraph.addEdge(module, stubModule);
      }
    }

    // execute
    try (Mutability mu = Mutability.create("Constellate")) {
      StarlarkThread thread = StarlarkThread.create(mu, semantics, "constellate", null);
      // We use the default print handler, which writes to stderr.
      thread.setLoader(imports::get);
      // Fake Bazel's "export" hack, by which provider symbols
      // bound to global variables take on the name of the global variable.
      thread.setPostAssignHook((name, location, value) -> {
        // Post assign hook now receives: String name, Location location, Object value
        if (value instanceof FakeProviderApi) {
          ((FakeProviderApi) value).setName(name);
        } else if (value instanceof FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) {
          FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier functionIdentifier = (FakeStarlarkRuleFunctionsApi.RuleDefinitionIdentifier) value;
          functionIdentifier.setAssignedName(name);
        }
      });
      Starlark.execFileProgram(prog, module, thread);
    } catch (EvalException ex) {
      throw new StarlarkEvaluationException(ex.getMessageWithStack());
    }

    pending.remove(label);
    loaded.put(label, module);

    // Best-effort enhancement: extract OriginKey and other metadata from real evaluated objects
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
      logger.atInfo().log("1 pathOfLabel %s: workspaceRoot=%s", label, workspaceRoot);
      return Paths.get(label.toPathFragment().toString());
    }
    if (label.getWorkspaceName().equals(workspaceName)) {
      logger.atInfo().log("2 pathOfLabel %s: workspaceRoot=%s", label, workspaceRoot);
      return Paths.get(label.toPathFragment().toString());
    }
    logger.atInfo().log("3 pathOfLabel %s: workspaceRoot=%s", label, workspaceRoot);
    return Paths.get(workspaceRoot, label.toPathFragment().toString());
  }

  public ParserInput getInputSource(String bzlWorkspacePath) throws IOException {
    for (String rootPath : depRoots) {
      String filepath = rootPath + "/" + bzlWorkspacePath;
      if (fileAccessor.fileExists(filepath)) {
        logger.atInfo().log("🟢 found input source %s (%s)", bzlWorkspacePath, filepath);
        return fileAccessor.inputSource(filepath);
      } else {
        logger.atInfo().log("🔴 file not found: %s (%s)", bzlWorkspacePath, filepath);
      }
    }

    logger.atWarning().log("getInputSource failed %s: %s", bzlWorkspacePath, depRoots);

    // All depRoots attempted and no valid file was found.
    throw new NoSuchFileException(bzlWorkspacePath);
  }

  // private static Binding asBinding(Entry<String, Object> envEntry) {
  // Binding.Builder binding = Binding.newBuilder();
  // return binding.build();
  // }

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
    }

    logger.atInfo().log("Created stub module for failed load '%s' with %d symbols: %s", loadLabel, symbols.size(),
        symbols);

    return stubModule;
  }

  private static void addMorePredeclared(ImmutableMap.Builder<String, Object> env) {
    // Add dummy declarations that would come from packages.StarlarkLibrary.COMMON
    // were Constellate allowed to depend on it. See hack for select below.
    env.put("json", Json.INSTANCE);
    env.put("proto", new ProtoModule());
    env.put("depset", new StarlarkCallable() {
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
      public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) throws EvalException {
        String message = positional.length > 0 ? Starlark.str(positional[0], thread.getSemantics()) : "fail() called";
        logger.atInfo().log("fail() called: %s", message);
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
}
