package com.google.devtools.build.lib.packages;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Implementation of --incompatible_autoload_externally.
 *
 * The
 */
public class AutoloadsConfiguration {

  private static final ImmutableSet<String> PREDECLARED_PRISTINE_REPOS = ImmutableSet.of(
      "rules_python", "bazel_skylib", "rules_android", "com_google_protobuf", "protobuf",
      "bazel_tools"
  );

  // Value of --incompatible_autoload_externally, symbols, possibly prefixed with +/-
  private final ImmutableList<String> symbolConfiguration;

  // Repositories where autoloads shouldn't be used
  private final ImmutableSet<String> pristineRepos;

  // bzl environment where autoloads are used, uninjected (not loaded yet)
  private final ImmutableMap<String, Object> uninjectedBuildBzlEnvWithAutoloads;

  // bzl environment where autoloads aren't used, uninjected (not loaded yet)
  private final ImmutableMap<String, Object> uninjectedBuildBzlEnvPristine;

  // Used for nicer error messages
  private final boolean bzlmodEnabled;

  public AutoloadsConfiguration(RuleClassProvider ruleClassProvider, StarlarkSemantics semantics) {
    symbolConfiguration = ImmutableList.copyOf(
        semantics.get(BuildLanguageOptions.INCOMPATIBLE_AUTOLOAD_EXTERNALLY));

    // Validates the inputs
    Set<String> uniqueSymbols = new HashSet<>();
    for (String symbol : semantics.get(BuildLanguageOptions.INCOMPATIBLE_AUTOLOAD_EXTERNALLY)) {
      String symbolWithoutPrefix =
          symbol.startsWith("+") || symbol.startsWith("-") ? symbol.substring(1) : symbol;
      if (!uniqueSymbols.add(symbolWithoutPrefix)) {
        throw new IllegalStateException(
            String.format("Duplicated symbol '%s' in --incompatible_autoload_externally",
                symbolWithoutPrefix));
      }
      if (!AUTOLOAD_CONFIG.containsKey(symbolWithoutPrefix)) {
        throw new IllegalStateException("Undefined symbol in --incompatible_autoload_externally");
      }
    }

    this.pristineRepos = ImmutableSet.<String>builder().addAll(PREDECLARED_PRISTINE_REPOS).addAll(
        semantics.get(BuildLanguageOptions.REPOSITORIES_WITHOUT_AUTOLOAD)).build();

    ImmutableMap<String, Object> originalBuildBzlEnv = ruleClassProvider
        .getBazelStarlarkEnvironment()
        .getUninjectedBuildBzlEnv();

    // Sets up environments for BzlCompile function
    this.uninjectedBuildBzlEnvWithAutoloads = modifyBuildBzlEnv(originalBuildBzlEnv,
        /* add= */filterSymbols(symbolConfiguration, symbol -> !symbol.startsWith("-")).stream()
            .collect(toImmutableMap(key -> key, key -> Starlark.NONE)),
        /* remove= */ filterSymbols(symbolConfiguration, symbol -> symbol.startsWith("-")));
    this.uninjectedBuildBzlEnvPristine = modifyBuildBzlEnv(originalBuildBzlEnv,
        /* add= */ ImmutableMap.of(),
        /* remove= */ filterSymbols(symbolConfiguration, symbol -> !symbol.startsWith("+")));

    // Validate rdeps - this ensures that all the rules using a provider are also removed
    // Check what's still available in Bazel (some symbols might already be deleted)
    ImmutableSet<String> allAvailableSymbols = ImmutableSet.<String>builder()
        .addAll(uninjectedBuildBzlEnvPristine.keySet()).addAll(convertNativeStructToMap(
            (StarlarkInfo) uninjectedBuildBzlEnvPristine.get("native")).keySet()).build();
    for (String symbol : filterSymbols(symbolConfiguration, symbol -> !symbol.startsWith("+"))) {
      ImmutableList<String> unsatisfiedRdeps = AUTOLOAD_CONFIG.get(symbol).getRdeps().stream()
          .filter(s -> allAvailableSymbols.contains(s)).collect(
              toImmutableList());
      if (!unsatisfiedRdeps.isEmpty()) {
        throw new IllegalStateException(
            String.format("Symbol in '%s' can't be removed, because it's still used by: %s", symbol,
                unsatisfiedRdeps.stream().collect(
                    Collectors.joining(", "))));
      }
    }

    this.bzlmodEnabled = semantics.getBool(BuildLanguageOptions.ENABLE_BZLMOD);
  }

  /**
   * Returns the environment for BzlCompile function
   */
  public ImmutableMap<String, Object> getUninjectedBuildBzlEnv(Label key) {
    return checkPristine(key) ? uninjectedBuildBzlEnvPristine : uninjectedBuildBzlEnvWithAutoloads;
  }

  /**
   * Check if autoloads shouldn't be used.
   */
  public boolean checkPristine(Label key) {
    return key == null || pristineRepos.contains(key.getRepository().getName());
  }

  /**
   * Modifies the environment for BzlLoad function (returned from StarlarkBuiltinsFunction)
   */
  public ImmutableMap<String, Object> modifyBuildBzlEnv(
      boolean isWithAutoloads,
      ImmutableMap<String, Object> originalEnv,
      ImmutableMap<String, Object> newSymbols) {
    if (isWithAutoloads) {
      return modifyBuildBzlEnv(originalEnv,
          /* add= */ newSymbols,
          /* remove= */ filterSymbols(symbolConfiguration, symbol -> symbol.startsWith("-")));
    } else {
      return modifyBuildBzlEnv(originalEnv,
          /* add= */ ImmutableMap.of(),
          /* remove= */ filterSymbols(symbolConfiguration, symbol -> !symbol.startsWith("+")));
    }
  }

  /**
   * Modifies the environment for Package function (returned from StarlarkBuiltinsFunction).
   */
  public ImmutableMap<String, Object> modifyBuildEnv(boolean isWithAutoloads,
      ImmutableMap<String, Object> originalEnv, ImmutableMap<String, Object> newSymbols) {
    final ImmutableMap<String, Object> add;
    // Sets up environments for BzlCompile function
    if (isWithAutoloads) {
      add = newSymbols;
    } else {
      add = ImmutableMap.of();
    }
    Map<String, Object> envBuilder = new LinkedHashMap<>(originalEnv);
    for (var symbol : add.entrySet()) {
      if (AutoloadsConfiguration.AUTOLOAD_CONFIG.get(symbol.getKey()).isRule()) {
        envBuilder.put(symbol.getKey(), symbol.getValue());
      }
    }
    ImmutableList<String> remove = filterSymbols(symbolConfiguration,
        symbol -> symbol.startsWith("-"));
    for (String symbol : remove) {
      if (AutoloadsConfiguration.AUTOLOAD_CONFIG.get(symbol).isRule()) {
        envBuilder.remove(symbol);
      }
    }
    return ImmutableMap.copyOf(envBuilder);
  }

  /**
   * Creates modified environment that's used in BzlCompileFunction and StarlarkBuiltinsFunction.
   *
   * It starts with the original environment. Adds the symbols to it or removes them.
   */
  private ImmutableMap<String, Object> modifyBuildBzlEnv(ImmutableMap<String, Object> originalEnv,
      ImmutableMap<String, Object> add,
      ImmutableList<String> remove) {
    Map<String, Object> envBuilder = new LinkedHashMap<>(originalEnv);
    Map<String, Object> nativeBindings = convertNativeStructToMap(
        (StarlarkInfo) envBuilder.remove("native"));

    for (var symbol : add.entrySet()) {
      if (AutoloadsConfiguration.AUTOLOAD_CONFIG.get(symbol.getKey()).isRule()) {
        nativeBindings.put(symbol.getKey(), symbol.getValue());
      } else {
        envBuilder.put(symbol.getKey(), symbol.getValue());
      }
    }
    for (String symbol : remove) {
      if (AutoloadsConfiguration.AUTOLOAD_CONFIG.get(symbol).isRule()) {
        nativeBindings.remove(symbol);
      } else {
        envBuilder.remove(symbol);
      }
    }
    envBuilder.put("native",
        StructProvider.STRUCT.create(nativeBindings, "no native function or rule '%s'"));
    return ImmutableMap.copyOf(envBuilder);
  }

  private static ImmutableList<String> filterSymbols(ImmutableList<String> symbols,
      Predicate<String> when) {
    return symbols.stream().filter(when).map(
            symbol -> symbol.startsWith("+") || symbol.startsWith("-") ? symbol.substring(1) : symbol)
        .collect(
            toImmutableList());
  }

  private static LinkedHashMap<String, Object> convertNativeStructToMap(StarlarkInfo struct) {
    LinkedHashMap<String, Object> destr = new LinkedHashMap<>();
    for (String field : struct.getFieldNames()) {
      destr.put(field, struct.getValue(field));
    }
    return destr;
  }

  /**
   * Returns a list of all the symbols that need to be loaded
   */
  public ImmutableList<BzlLoadValue.Key> getAutoloads() {
    ImmutableList<String> symbolsToLoad =
        filterSymbols(symbolConfiguration, s -> !s.startsWith("-"));

    // Inject loads for rules and symbols removed from Bazel
    ImmutableList.Builder<BzlLoadValue.Key> loadKeysBuilder = ImmutableList.builder();
    for (String symbol : symbolsToLoad) {
      loadKeysBuilder.add(AutoloadsConfiguration.AUTOLOAD_CONFIG.get(symbol).getKey());
    }
    return loadKeysBuilder.build();
  }

  /**
   * Processes LoadedValues into a map of symbols
   */
  public ImmutableMap<String, Object> processLoads(BzlLoadValue[] autoloadValues)
      throws AutoloadException {
    if (autoloadValues.length == 0) {
      return ImmutableMap.of();
    }
    ImmutableMap.Builder<String, Object> newSymbols = ImmutableMap.builder();

    ImmutableList<String> symbolsToLoad =
        filterSymbols(symbolConfiguration, s -> !s.startsWith("-"));
    int i = 0;
    String workspaceWarning =
        bzlmodEnabled
            ? ""
            : " Most likely you need to upgrade the version of rules repository providing it your"
                + " WORKSPACE file.";
    for (String symbol : symbolsToLoad) {
      // Check if the symbol is named differently in the bzl file than natively. Renames are rare:
      // Example is renaming native.ProguardSpecProvider to ProguardSpecInfo.
      String newName = AUTOLOAD_CONFIG.get(symbol).getNewName();
      if (newName == null) {
        newName = symbol;
      }
      BzlLoadValue v = autoloadValues[i];
      Object symbolValue = v.getModule().getGlobal(newName);
      if (symbolValue == null) {
        throw new AutoloadException(
            String.format(
                "The toplevel symbol '%s' set by --incompatible_load_symbols_externally couldn't"
                    + " be loaded. '%s' not found in auto loaded '%s'.%s",
                symbol, workspaceWarning, newName, AUTOLOAD_CONFIG.get(symbol).getLoadLabel()));
      }
      newSymbols.put(symbol, symbolValue); // Exposed as old name
    }
    return newSymbols.buildOrThrow();
  }

  @AutoValue
  public abstract static class SymbolRedirect {

    @Nullable
    public abstract Label getLoadLabel();

    public abstract boolean isRule();

    @Nullable
    public abstract String getNewName();

    public abstract ImmutableSet<String> getRdeps();

    public BzlLoadValue.Key getKey() {
      return BzlLoadValue.keyForBuild(getLoadLabel());
    }
  }


  /**
   * Indicates a problem performing builtins injection.
   */
  public static final class AutoloadException extends Exception {

    AutoloadException(String message) {
      super(message);
    }
  }

  private static SymbolRedirect configRule(String label) {
    return new AutoValue_AutoloadsConfiguration_SymbolRedirect(Label.parseCanonicalUnchecked(label),
        true, null, ImmutableSet.of());
  }

  private static SymbolRedirect configSymbol(String label, String... rdeps) {
    return new AutoValue_AutoloadsConfiguration_SymbolRedirect(
        label == null ? null : Label.parseCanonicalUnchecked(label),
        false, null, ImmutableSet.copyOf(rdeps));
  }

  private static SymbolRedirect configRenamedSymbol(String label, String newName,
      String... rdeps) {
    return new AutoValue_AutoloadsConfiguration_SymbolRedirect(Label.parseCanonicalUnchecked(label),
        false, newName, ImmutableSet.copyOf(rdeps));
  }

  private static final String[] androidRules = {"aar_import", "android_binary", "android_library",
      "android_local_test", "android_sdk"};

  public static final ImmutableMap<String, SymbolRedirect> AUTOLOAD_CONFIG = ImmutableMap.<String, SymbolRedirect>builder()
      .put("CcSharedLibraryInfo",
          configSymbol("@rules_cc//cc/common:cc_shared_library_info.bzl", "cc_shared_library"))
      .put("CcSharedLibraryHintInfo",
          configSymbol("@rules_cc//cc/common:cc_shared_library_hint_info.bzl", "cc_common"))
      .put("cc_proto_aspect",
          configSymbol("@com_google_protobuf//bazel/common:cc_proto_library.bzl",
              "cc_proto_library"))
      .put("ProtoInfo",
          configSymbol("@com_google_protobuf//bazel/common:proto_info.bzl", "proto_library",
              "cc_proto_library", "cc_shared_library", "java_lite_proto_library",
              "java_proto_library", "proto_lang_toolchain", "java_binary", "py_extension",
              "proto_common_do_not_use"))
      .put("proto_common_do_not_use",
          configSymbol("@com_google_protobuf//bazel/common:proto_common.bzl"))
      .put("cc_common", configSymbol("@rules_cc//cc/common:cc_common.bzl"))
      .put("CcInfo",
          configSymbol("@rules_cc//cc/common:cc_info.bzl", "cc_binary", "cc_library", "cc_test",
              "cc_shared_library", "cc_common", "java_library", "cc_proto_library", "java_import",
              "java_runtime", "java_binary", "objc_library", "java_common", "JavaInfo",
              "py_extension", "cc_import", "objc_import", "objc_library", "cc_toolchain",
              "PyCcLinkParamsProvider", "py_library"))
      .put("DebugPackageInfo",
          configSymbol("@rules_cc//cc/common:debug_package_info.bzl", "cc_binary", "cc_test"))
      .put("CcToolchainConfigInfo",
          configSymbol("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "cc_toolchain"))
      .put("java_common", configSymbol("@rules_java//java/common:java_common.bzl"))
      .put("JavaInfo",
          configSymbol("@rules_java//java/common:java_info.bzl", "java_binary", "java_library",
              "java_test", "java_proto_library", "java_lite_proto_library", "java_plugin",
              "java_import", "java_common"))
      .put("JavaPluginInfo",
          configSymbol("@rules_java//java/common:java_plugin_info.bzl", "java_plugin",
              "java_library", "java_binary", "java_test"))
      .put("ProguardSpecProvider",
          configRenamedSymbol("@rules_java//java/common:proguard_spec_info.bzl",
              "ProguardSpecInfo", "java_lite_proto_library", "java_import", "android_binary",
              "android_library"))
      .put("android_common", configSymbol("@rules_android//rules:common.bzl"))
      .put("AndroidIdeInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("ApkInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidInstrumentationInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidResourcesInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidNativeLibsInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidApplicationResourceInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidBinaryNativeLibsInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidSdkInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidManifestInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidAssetsInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidLibraryAarInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidProguardInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidIdlInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidPreDexJarInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidCcLinkParamsInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("DataBindingV2Info", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidLibraryResourceClassJarProvider",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidFeatureFlagSet",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("ProguardMappingInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidBinaryData", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("BaselineProfileProvider",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidNeverLinkLibrariesProvider",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidOptimizedJarInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidDexInfo", configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("AndroidOptimizationInfo",
          configSymbol("@rules_android//rules:providers.bzl", androidRules))
      .put("PyInfo",
          configSymbol("@rules_python//python:py_info.bzl", "py_binary", "py_test", "py_library"))
      .put("PyRuntimeInfo",
          configSymbol("@rules_python//python:py_runtime_info.bzl", "py_binary", "py_test",
              "py_library"))
      .put("PyCcLinkParamsProvider",
          configRenamedSymbol("@rules_python//python:py_cc_link_params_info.bzl",
              "PyCcLinkParamsInfo", "py_binary", "py_test", "py_library"))
      .put("aar_import", configSymbol("@rules_android//rules:rules.bzl"))
      .put("android_binary", configRule("@rules_android//rules:rules.bzl"))
      .put("android_device_script_fixture",
          configRule("@rules_android//rules:rules.bzl"))
      .put("android_host_service_fixture",
          configRule("@rules_android//rules:rules.bzl"))
      .put("android_library", configRule("@rules_android//rules:rules.bzl"))
      .put("android_local_test", configRule("@rules_android//rules:rules.bzl"))
      .put("android_sdk", configRule("@rules_android//rules:rules.bzl"))
      .put("android_tools_defaults_jar", configRule("@rules_android//rules:rules.bzl"))
      .put("cc_binary", configRule("@rules_cc//cc:cc_binary.bzl"))
      .put("cc_import", configRule("@rules_cc//cc:cc_import.bzl"))
      .put("cc_library", configRule("@rules_cc//cc:cc_library.bzl"))
      .put("cc_proto_library",
          configRule("@com_google_protobuf//bazel:cc_proto_library.bzl"))
      .put("cc_shared_library", configRule("@rules_cc//cc:cc_shared_library.bzl"))
      .put("cc_test", configRule("@rules_cc//cc:cc_test.bzl"))
      .put("cc_toolchain", configRule("@rules_cc//cc/toolchains:cc_toolchain.bzl"))
      .put("cc_toolchain_suite",
          configRule("@rules_cc//cc/toolchains:cc_toolchain_suite.bzl"))
      .put("fdo_prefetch_hints",
          configRule("@rules_cc//cc/toolchains:fdo_prefetch_hints.bzl"))
      .put("fdo_profile", configRule("@rules_cc//cc/toolchains:fdo_profile.bzl"))
      .put("java_binary", configRule("@rules_java//java:java_binary.bzl"))
      .put("java_import", configRule("@rules_java//java:java_import.bzl"))
      .put("java_library", configRule("@rules_java//java:java_library.bzl"))
      .put("java_lite_proto_library",
          configRule("@com_google_protobuf//bazel:java_lite_proto_library.bzl"))
      .put("java_package_configRuleuration",
          configRule("@rules_java//java/toolchains:java_package_configRuleuration.bzl"))
      .put("java_plugin", configRule("@rules_java//java:java_plugin.bzl"))
      .put("java_proto_library",
          configRule("@com_google_protobuf//bazel:java_proto_library.bzl"))
      .put("java_runtime", configRule("@rules_java//java/toolchains:java_runtime.bzl"))
      .put("java_test", configRule("@rules_java//java:java_test.bzl"))
      .put("java_toolchain",
          configRule("@rules_java//java/toolchains:java_toolchain.bzl"))
      .put("memprof_profile",
          configRule("@rules_cc//cc/toolchains:memprof_profile.bzl"))
      .put("objc_import", configRule("@rules_cc//cc:objc_import.bzl"))
      .put("objc_library", configRule("@rules_cc//cc:objc_library.bzl"))
      .put("propeller_optimize",
          configRule("@rules_cc//cc/toolchains:propeller_optimize.bzl"))
      .put("proto_lang_toolchain",
          configRule("@com_google_protobuf//bazel/toolchain:proto_lang_toolchain.bzl"))
      .put("proto_library", configRule("@com_google_protobuf//bazel:proto_library.bzl"))
      .put("py_binary", configRule("@rules_python//python:py_binary.bzl"))
      .put("py_library", configRule("@rules_python//python:py_library.bzl"))
      .put("py_runtime", configRule("@rules_python//python:py_runtime.bzl"))
      .put("py_test", configRule("@rules_python//python:py_test.bzl"))
      .put("sh_binary", configRule("@rules_sh//sh:sh_binary.bzl"))
      .put("sh_library", configRule("@rules_sh//sh:sh_library.bzl"))
      .put("sh_test", configRule("@rules_sh//sh:sh_test.bzl"))
      .put("available_xcodes", configRule("@apple_support//xcode:available_xcodes.bzl"))
      .put("xcode_config", configRule("@apple_support//xcode:xcode_configRule.bzl"))
      .put("xcode_config_alias",
          configRule("@apple_support//xcode:xcode_configRule_alias.bzl"))
      .put("xcode_version", configRule("@apple_support//xcode:xcode_version.bzl"))
      .build();

  // TODO: figure out apple_common
}
