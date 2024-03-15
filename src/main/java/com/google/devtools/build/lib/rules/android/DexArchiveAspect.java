// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType.UNQUOTED;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.StarlarkProviderIdentifier.forKey;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.rules.android.AndroidCommon.getAndroidConfig;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** Aspect to {@link DexArchiveProvider build .dex Archives} from Jars. */
public class DexArchiveAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "DexArchiveAspect";

  /**
   * Function that returns a {@link Rule}'s {@code incremental_dexing} attribute for use by this
   * aspect. Must be provided when attaching this aspect to a target.
   */
  @SerializationConstant
  public static final Function<Rule, AspectParameters> PARAM_EXTRACTOR =
      (Rule rule) -> {
        AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
        AspectParameters.Builder result = new AspectParameters.Builder();
        TriState incrementalAttr = attributes.get("incremental_dexing", TRISTATE);
        result.addAttribute("incremental_dexing", incrementalAttr.name());
        result.addAttribute(
            "min_sdk_version", attributes.get("min_sdk_version", INTEGER).toString());
        result.addAttribute("toprule_kind", rule.getRuleClass());
        return result.build();
      };

  /**
   * Function that limits this aspect to Java 8 desugaring (disabling incremental dexing) when
   * attaching this aspect to a target. This is intended for implicit attributes like the stub APKs
   * for {@code bazel mobile-install}.
   */
  @SerializationConstant
  static final Function<Rule, AspectParameters> ONLY_DESUGAR_JAVA8 =
      (Rule rule) ->
          new AspectParameters.Builder()
              .addAttribute("incremental_dexing", TriState.NO.name())
              .build();
  /** Aspect-only label for dexbuidler executable, to avoid name clashes with labels on rules. */
  private static final String ASPECT_DEXBUILDER_PREREQ = "$dex_archive_dexbuilder";
  /** Aspect-only label for desugaring executable, to avoid name clashes with labels on rules. */
  private static final String ASPECT_DESUGAR_PREREQ = "$aspect_desugar";

  private static final ImmutableList<String> TRANSITIVE_ATTRIBUTES =
      ImmutableList.of(
          "deps",
          "exports",
          "runtime_deps",
          // Propagate the aspect down legacy toolchain dependencies. This won't work for platform-
          // based toolchains, which aren't connected to an attribute. See
          // propagateDownLegacyToolchain for how this distinction is handled.
          ":android_sdk",
          "aidl_lib", // for the aidl runtime in the android_sdk rule
          "$toolchain", // this is _toolchain in Starlark rules (b/78647825)
          "$build_stamp_deps", // for build stamp runtime class deps
          "$build_stamp_mergee_manifest_lib", // for empty build stamp Service class implementation
          // To get from proto_library through proto_lang_toolchain rule to proto runtime library.
          ":aspect_proto_toolchain_for_javalite",
          "runtime");

  private static final FlagMatcher DEXOPTS_SUPPORTED_IN_DEXBUILDER =
      new FlagMatcher(
          ImmutableList.of("--no-locals", "--no-optimize", "--no-warnings", "--positions"));

  private final RepositoryName toolsRepository;
  private final String sdkToolchainLabel;

  public DexArchiveAspect(RepositoryName toolsRepository, String sdkToolchainLabel) {
    this.toolsRepository = toolsRepository;
    this.sdkToolchainLabel = sdkToolchainLabel;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters params) {
    Label toolchainType = Label.parseCanonicalUnchecked(toolsRepository + sdkToolchainLabel);
    AspectDefinition.Builder result =
        new AspectDefinition.Builder(this)
            .requireStarlarkProviders(forKey(JavaInfo.PROVIDER.getKey()))
            // Latch onto Starlark toolchains in case they have a "runtime" (b/78647825)
            .requireStarlarkProviders(forKey(ToolchainInfo.PROVIDER.getKey()))
            // For android_sdk rules, where we just want to get at aidl runtime deps.
            .requireStarlarkProviders(forKey(AndroidSdkProvider.PROVIDER.getKey()))
            .requireStarlarkProviders(forKey(ProtoInfo.PROVIDER.getKey()))
            .requireStarlarkProviderSets(
                ImmutableList.of(
                    // For proto_lang_toolchain rules, where we just want to get at their runtime
                    // deps.
                    ImmutableSet.of(ProtoLangToolchainProvider.PROVIDER_ID)))
            .addToolchainTypes(
                ToolchainTypeRequirement.builder(toolchainType).mandatory(true).build())
            // Parse labels since we don't have RuleDefinitionEnvironment.getLabel like in a rule
            .add(
                attr(ASPECT_DESUGAR_PREREQ, LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .exec()
                    .value(
                        Label.parseCanonicalUnchecked(
                            toolsRepository + "//tools/android:desugar_java8")))
            // Access to --android_sdk so we can stub in a bootclasspath for desugaring if missing
            // Remove this entirely when we remove --android_sdk support.
            .add(
                attr(":dex_archive_android_sdk", LABEL)
                    .allowedRuleClasses("android_sdk", "filegroup")
                    .value(
                        AndroidRuleClasses.getAndroidSdkLabel(
                            Label.parseCanonicalUnchecked(
                                toolsRepository + AndroidRuleClasses.DEFAULT_SDK))))
            .add(
                Allowlist.getAttributeFromAllowlistName("enable_starlark_dex_desugar_proguard")
                    .value(
                        Label.parseCanonicalUnchecked(
                            toolsRepository
                                + "//tools/allowlists/android_binary_allowlist:enable_starlark_dex_desugar_proguard")))
            .requiresConfigurationFragments(AndroidConfiguration.class)
            .requireAspectsWithProviders(
                ImmutableList.of(ImmutableSet.of(forKey(JavaInfo.PROVIDER.getKey()))));
    if (TriState.valueOf(params.getOnlyValueOfAttribute("incremental_dexing")) != TriState.NO) {
      // Marginally improves "query2" precision for targets that disable incremental dexing
      result.add(
          attr(ASPECT_DEXBUILDER_PREREQ, LABEL)
              .cfg(ExecutionTransitionFactory.createFactory())
              .exec()
              .value(
                  Label.parseCanonicalUnchecked(toolsRepository + "//tools/android:dexbuilder")));
    }
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      result.propagateAlongAttribute(attr);
    }
    return result.build();
  }

  /**
   * Returns toolchain .jars that need dexing for platform-based toolchains.
   *
   * <p>Legacy toolchains handle these .jars recursively by propagating the aspect down the
   * ":android_sdk" attribute. So they don't need this method.
   */
  private static ImmutableList<Artifact> getPlatformBasedToolchainJars(RuleContext ruleContext)
      throws RuleErrorException {
    if (!ruleContext.attributes().has(":android_sdk")) {
      // If we're dexing a non-Android target (like a java_library), there's no Android toolchain to
      // include.
      return ImmutableList.of();
    }

    AndroidSdkProvider androidSdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    if (androidSdk == null || androidSdk.getAidlLib() == null) {
      // If the Android SDK is null, we don't have a valid toolchain. Expect a rule error reported
      // from AndroidSdkProvider.
      return ImmutableList.of();
    }
    return ImmutableList.copyOf(
        JavaInfo.getJavaInfo(androidSdk.getAidlLib()).getDirectRuntimeJars());
  }

  @Override
  @Nullable
  public ConfiguredAspect create(
      Label targetLabel,
      ConfiguredTarget ct,
      RuleContext ruleContext,
      AspectParameters params,
      RepositoryName toolsRepository)
      throws InterruptedException, ActionConflictException {
    ConfiguredAspect.Builder result = new ConfiguredAspect.Builder(ruleContext);

    // No-op out of the aspect in the android_binary rule if the Starlark dex/desugar will execute
    // to avoid registering duplicate actions and bloating memory.
    if (!params.getAttribute("toprule_kind").isEmpty()
        && params.getOnlyValueOfAttribute("toprule_kind").equals("android_binary")
        && Allowlist.hasAllowlist(ruleContext, "enable_starlark_dex_desugar_proguard")
        && Allowlist.isAvailable(ruleContext, "enable_starlark_dex_desugar_proguard")) {
      return result.build();
    }

    int minSdkVersion = 0;
    if (!params.getAttribute("min_sdk_version").isEmpty()) {
      minSdkVersion = Integer.valueOf(params.getOnlyValueOfAttribute("min_sdk_version"));
    }

    Function<Artifact, Artifact> desugaredJars;
    ImmutableList<Artifact> extraToolchainJars;
    ImmutableCollection<Artifact> runtimeJars;
    try {
      extraToolchainJars = getPlatformBasedToolchainJars(ruleContext);
      desugaredJars =
          desugarJarsIfRequested(ct, ruleContext, minSdkVersion, result, extraToolchainJars);
      runtimeJars = getProducedRuntimeJars(ct, ruleContext, extraToolchainJars);
    } catch (RuleErrorException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    }

    TriState incrementalAttr =
        TriState.valueOf(params.getOnlyValueOfAttribute("incremental_dexing"));
    if (incrementalAttr == TriState.NO
        || (!getAndroidConfig(ruleContext).useIncrementalDexing()
            && incrementalAttr == TriState.AUTO)) {
      // Dex archives will never be used, so don't bother setting them up.
      return result.build();
    }

    if (JavaCommon.isNeverLink(ruleContext)) {
      return result.addProvider(DexArchiveProvider.NEVERLINK).build();
    }

    DexArchiveProvider.Builder dexArchives = new DexArchiveProvider.Builder();
    collectPrerequisites(
        ruleContext, DexArchiveProvider.class, dexArchives::addTransitiveProviders);
    if (runtimeJars != null) {
      boolean basenameClash = checkBasenameClash(runtimeJars);
      Set<Set<String>> aspectDexopts = aspectDexopts(ruleContext);
      String minSdkFilenamePart = minSdkVersion > 0 ? "--min_sdk_version=" + minSdkVersion : "";
      for (Artifact jar : runtimeJars) {
        for (Set<String> incrementalDexopts : aspectDexopts) {
          // Since we're potentially dexing the same jar multiple times with different flags, we
          // need to write unique artifacts for each flag combination. Here, it is convenient to
          // distinguish them by putting the flags that were used for creating the artifacts into
          // their filenames. Since min_sdk_version is a parameter to the aspect from the
          // android_binary target that the aspect originates from, it's handled separately so that
          // the correct min sdk value is used.
          String uniqueFilename =
              (basenameClash ? jar.getRootRelativePathString() : jar.getFilename())
                  + Joiner.on("").join(incrementalDexopts)
                  + minSdkFilenamePart
                  + ".dex.zip";
          Artifact dexArchive =
              createDexArchiveAction(
                  ruleContext,
                  ASPECT_DEXBUILDER_PREREQ,
                  desugaredJars.apply(jar),
                  incrementalDexopts,
                  minSdkVersion,
                  AndroidBinary.getDxArtifact(ruleContext, uniqueFilename));
          dexArchives.addDexArchive(incrementalDexopts, dexArchive, jar);
        }
      }
    }
    return result.addProvider(dexArchives.build()).build();
  }

  /**
   * Runs Jars in {@link JavaInfo#getDirectRuntimeJars()} through desugaring action if flag is set
   * and adds the result to {@code result}. Note that this cannot happen in a separate aspect
   * because aspects don't see providers added by other aspects executed on the same target.
   */
  private Function<Artifact, Artifact> desugarJarsIfRequested(
      ConfiguredTarget base,
      RuleContext ruleContext,
      int minSdkVersion,
      ConfiguredAspect.Builder result,
      Iterable<Artifact> extraToolchainJars)
      throws RuleErrorException {
    if (!getAndroidConfig(ruleContext).desugarJava8()) {
      return Functions.identity();
    }
    Map<Artifact, Artifact> newlyDesugared = new HashMap<>();
    if (JavaCommon.isNeverLink(ruleContext)) {
      result.addProvider(AndroidRuntimeJarProvider.NEVERLINK);
      return Functions.forMap(newlyDesugared);
    }
    AndroidRuntimeJarProvider.Builder desugaredJars = new AndroidRuntimeJarProvider.Builder();
    collectPrerequisites(
        ruleContext, AndroidRuntimeJarProvider.class, desugaredJars::addTransitiveProviders);
    if (isProtoLibrary(ruleContext)) {
      // TODO(b/33557068): Desugar protos if needed instead of assuming they don't need desugaring
      result.addProvider(desugaredJars.build());
      return Functions.identity();
    }

    JavaInfo javaInfo = JavaInfo.getJavaInfo(base);
    if (javaInfo != null) {
      // These are all transitive hjars of dependencies and hjar of the jar itself
      NestedSet<Artifact> compileTimeClasspath = JavaInfo.transitiveCompileTimeJars(base);
      ImmutableSet.Builder<Artifact> jars = ImmutableSet.builder();
      jars.addAll(javaInfo.getDirectRuntimeJars());

      Artifact rJar = getAndroidLibraryRJar(base);
      if (rJar != null) {
        // TODO(b/124540821): Disable R.jar desugaring (with a flag).
        jars.add(rJar);
      }

      Artifact buildStampJar = getAndroidBuildStampJar(base);
      if (buildStampJar != null) {
        jars.add(buildStampJar);
      }

      // For android_* targets we need to honor their bootclasspath (nicer in general to do so)
      NestedSet<Artifact> bootclasspath = getBootclasspath(base, ruleContext);

      jars.addAll(extraToolchainJars);
      ImmutableSet<Artifact> jarsToProcess = jars.build();
      boolean basenameClash = checkBasenameClash(jarsToProcess);
      for (Artifact jar : jarsToProcess) {
        Artifact desugared =
            createDesugarAction(
                ruleContext,
                basenameClash,
                jar,
                bootclasspath,
                compileTimeClasspath,
                minSdkVersion);
        newlyDesugared.put(jar, desugared);
        desugaredJars.addDesugaredJar(jar, desugared);
      }
    }
    result.addProvider(desugaredJars.build());
    return Functions.forMap(newlyDesugared);
  }

  @Nullable
  private static ImmutableCollection<Artifact> getProducedRuntimeJars(
      ConfiguredTarget base, RuleContext ruleContext, Iterable<Artifact> extraToolchainJars)
      throws RuleErrorException {
    if (isProtoLibrary(ruleContext)) {
      if (!ruleContext.getPrerequisites("srcs").isEmpty()) {
        JavaInfo javaInfo = JavaInfo.getJavaInfo(base);
        if (javaInfo != null) {
          return javaInfo.getJavaOutputs().stream()
              .map(JavaOutput::getClassJar)
              .collect(toImmutableList());
        }
      }
    } else {
      ImmutableSet.Builder<Artifact> jars = ImmutableSet.builder();
      JavaInfo javaInfo = JavaInfo.getJavaInfo(base);
      if (javaInfo != null) {
        jars.addAll(javaInfo.getDirectRuntimeJars());
      }

      Artifact rJar = getAndroidLibraryRJar(base);
      if (rJar != null) {
        jars.add(rJar);
      }

      Artifact buildStampJar = getAndroidBuildStampJar(base);
      if (buildStampJar != null) {
        jars.add(buildStampJar);
      }

      jars.addAll(extraToolchainJars);
      return jars.build();
    }
    return null;
  }

  private static boolean isProtoLibrary(RuleContext ruleContext) {
    return "proto_library".equals(ruleContext.getRule().getRuleClass());
  }

  @Nullable
  private static Artifact getAndroidLibraryRJar(ConfiguredTarget base) {
    AndroidIdeInfoProvider provider =
        (AndroidIdeInfoProvider) base.get(AndroidIdeInfoProvider.PROVIDER.getKey());
    if (provider != null && provider.getResourceJarJavaOutput() != null) {
      return provider.getResourceJarJavaOutput().getClassJar();
    }
    return null;
  }

  @Nullable
  private static Artifact getAndroidBuildStampJar(ConfiguredTarget base) {
    AndroidApplicationResourceInfo provider =
        (AndroidApplicationResourceInfo) base.get(AndroidApplicationResourceInfo.PROVIDER.getKey());
    if (provider != null && provider.getBuildStampJar() != null) {
      return provider.getBuildStampJar();
    }
    return null;
  }

  private static boolean checkBasenameClash(Iterable<Artifact> artifacts) {
    HashSet<String> seen = new HashSet<>();
    for (Artifact artifact : artifacts) {
      if (!seen.add(artifact.getFilename())) {
        return true;
      }
    }
    return false;
  }

  private static <T extends TransitiveInfoProvider> void collectPrerequisites(
      RuleContext ruleContext, Class<T> classType, Consumer<List<T>> sink) {
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (ruleContext.attributes().getAttributeType(attr) != null) {
        sink.accept(ruleContext.getPrerequisites(attr, classType));
      }
    }
  }

  private NestedSet<Artifact> getBootclasspath(ConfiguredTarget base, RuleContext ruleContext)
      throws RuleErrorException {
    NestedSet<Artifact> bootClasspath = JavaInfo.bootClasspath(base);
    if (!bootClasspath.isEmpty()) {
      return bootClasspath;
    }
    Artifact androidJar = getAndroidJar(ruleContext);
    if (androidJar != null) {
      return NestedSetBuilder.<Artifact>naiveLinkOrder().add(androidJar).build();
    }
    // This shouldn't ever be reached, but if it is, we should be clear about the error.
    throw new IllegalStateException("no compilationInfo or androidJar");
  }

  @Nullable
  private Artifact getAndroidJar(RuleContext ruleContext) throws RuleErrorException {
    Label toolchainType = Label.parseCanonicalUnchecked(toolsRepository + sdkToolchainLabel);
    AndroidSdkProvider androidSdk = AndroidSdkProvider.fromRuleContext(ruleContext, toolchainType);
    if (androidSdk == null) {
      // If the Android SDK is null, we don't have a valid toolchain. Expect a rule error reported
      // from AndroidSdkProvider.
      return null;
    }
    return androidSdk.getAndroidJar();
  }

  private Artifact createDesugarAction(
      RuleContext ruleContext,
      boolean disambiguateBasenames,
      Artifact jar,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> compileTimeClasspath,
      int minSdkVersion) {

    String minSdkFilenamePart = minSdkVersion > 0 ? "_minsdk=" + minSdkVersion : "";
    return createDesugarAction(
        ruleContext,
        ASPECT_DESUGAR_PREREQ,
        jar,
        bootclasspath,
        compileTimeClasspath,
        minSdkVersion,
        AndroidBinary.getDxArtifact(
            ruleContext,
            (disambiguateBasenames ? jar.getRootRelativePathString() : jar.getFilename())
                + minSdkFilenamePart
                + "_desugared.jar"));
  }

  private static Artifact createDesugarAction(
      RuleContext ruleContext,
      String desugarPrereqName,
      Artifact jar,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> classpath,
      int minSdkVersion,
      Artifact result) {
    SpawnAction.Builder action =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite(desugarPrereqName))
            .addInput(jar)
            .addTransitiveInputs(bootclasspath)
            .addTransitiveInputs(classpath)
            .addOutput(result)
            .setMnemonic("Desugar")
            .setProgressMessage("Desugaring %s for Android", jar.prettyPrint())
            .setExecutionInfo(
                createDexingDesugaringExecRequirements(ruleContext)
                    .putAll(ExecutionRequirements.WORKER_MODE_ENABLED)
                    .buildKeepingLast());

    CustomCommandLine.Builder args =
        new CustomCommandLine.Builder()
            .addExecPath("--input", jar)
            .addExecPath("--output", result)
            .addExecPaths(VectorArg.addBefore("--classpath_entry").each(classpath))
            .addExecPaths(VectorArg.addBefore("--bootclasspath_entry").each(bootclasspath));
    if (getAndroidConfig(ruleContext).checkDesugarDeps()) {
      args.add("--emit_dependency_metadata_as_needed");
    }
    if (getAndroidConfig(ruleContext).desugarJava8Libs()) {
      args.add("--desugar_supported_core_libs");
    }
    if (minSdkVersion > 0) {
      args.add("--min_sdk_version", Integer.toString(minSdkVersion));
    }

    action.addCommandLine(
        // Always use params file, so we don't need to compute command line length first
        args.build(), ParamFileInfo.builder(UNQUOTED).setUseAlways(true).build());

    ruleContext.registerAction(action.build(ruleContext));
    return result;
  }

  /**
   * Desugars the given Jar using an executable prerequisite {@code "$desugar"}. Rules calling this
   * method must declare the appropriate prerequisite, similar to how {@link #getDefinition} does it
   * for {@link DexArchiveAspect} under a different name.
   *
   * <p>It's useful to have this action separately since callers need to look up classpath and
   * bootclasspath in a different way than this aspect does it.
   *
   * @return the artifact given as {@code result}, which can simplify calling code
   */
  static Artifact desugar(
      RuleContext ruleContext,
      Artifact jar,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> classpath,
      int minSdkVersion,
      Artifact result) {
    return createDesugarAction(
        ruleContext, "$desugar", jar, bootclasspath, classpath, minSdkVersion, result);
  }

  /**
   * Creates a dexbuilder action with the given input, output, and flags. Flags must have been
   * filtered and normalized to a set that the dexbuilder tool can understand.
   *
   * @return the artifact given as {@code result}, which can simplify calling code
   */
  // Package-private method for use in AndroidBinary
  @CanIgnoreReturnValue
  static Artifact createDexArchiveAction(
      RuleContext ruleContext,
      String dexbuilderPrereq,
      Artifact jar,
      Set<String> incrementalDexopts,
      int minSdkVersion,
      Artifact dexArchive) {
    SpawnAction.Builder dexbuilder =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite(dexbuilderPrereq))
            .setExecutionInfo(
                createDexingDesugaringExecRequirements(ruleContext)
                    .putAll(
                        TargetUtils.getExecutionInfo(
                            ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
                    .buildKeepingLast())
            // WorkerSpawnStrategy expects the last argument to be @paramfile
            .addInput(jar)
            .addOutput(dexArchive)
            .setMnemonic("DexBuilder")
            .setProgressMessage(
                "Dexing %s with applicable dexopts %s", jar.prettyPrint(), incrementalDexopts);

    CustomCommandLine.Builder args =
        new CustomCommandLine.Builder()
            .addExecPath("--input_jar", jar)
            .addExecPath("--output_zip", dexArchive)
            .addAll(ImmutableList.copyOf(incrementalDexopts));
    if (minSdkVersion > 0) {
      args.add("--min_sdk_version", Integer.toString(minSdkVersion));
    }

    dexbuilder.addCommandLine(
        args.build(), ParamFileInfo.builder(UNQUOTED).setUseAlways(true).build());
    ruleContext.registerAction(dexbuilder.build(ruleContext));
    return dexArchive;
  }

  @CanIgnoreReturnValue
  static Artifact createShardedOptimizedDexArchiveAction(
      RuleContext ruleContext,
      Artifact jar,
      Set<String> incrementalDexopts,
      int minSdkVersion,
      Artifact dexArchive) {
    SpawnAction.Builder dexbuilder =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite(":optimizing_dexer"))
            .setExecutionInfo(
                createDexingDesugaringExecRequirements(ruleContext)
                    .putAll(
                        TargetUtils.getExecutionInfo(
                            ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
                    .buildKeepingLast())
            .addInput(jar)
            .addOutput(dexArchive)
            .setMnemonic("ShardedOptimizingDex")
            .setProgressMessage(
                "Optimized dexing %s with applicable dexopts %s",
                jar.prettyPrint(), incrementalDexopts);

    CustomCommandLine.Builder args =
        new CustomCommandLine.Builder()
            .add("--intermediate")
            .add("--release")
            .add("--no-desugaring")
            .addExecPath("--output", dexArchive)
            .addAll(ImmutableList.copyOf(incrementalDexopts))
            .addExecPath(jar);
    if (minSdkVersion > 0) {
      args.add("--min_sdk_version", Integer.toString(minSdkVersion));
    }

    dexbuilder.addCommandLine(args.build());
    ruleContext.registerAction(dexbuilder.build(ruleContext));
    return dexArchive;
  }

  private static Set<Set<String>> aspectDexopts(RuleContext ruleContext) {
    return Sets.powerSet(
        normalizeDexopts(getAndroidConfig(ruleContext).getDexoptsSupportedInIncrementalDexing()));
  }

  /** Creates the execution requires for the DexBuilder and Desugar actions */
  private static ImmutableMap.Builder<String, String> createDexingDesugaringExecRequirements(
      RuleContext ruleContext) {
    final ImmutableMap.Builder<String, String> executionInfo = ImmutableMap.builder();
    AndroidConfiguration androidConfiguration = getAndroidConfig(ruleContext);
    if (androidConfiguration.persistentDexDesugar()) {
      executionInfo.putAll(ExecutionRequirements.WORKER_MODE_ENABLED);
      if (androidConfiguration.persistentMultiplexDexDesugar()) {
        executionInfo.putAll(ExecutionRequirements.WORKER_MULTIPLEX_MODE_ENABLED);
      }
    }

    return executionInfo;
  }

  /**
   * Derives options to use in incremental dexing actions from the given context and dx flags, where
   * the latter typically come from a {@code dexopts} attribute on a top-level target. This method
   * only works reliably if the given dexopts were tokenized, e.g., using {@link
   * RuleContext#getTokenizedStringListAttr}.
   */
  static ImmutableSet<String> incrementalDexopts(
      RuleContext ruleContext, Iterable<String> tokenizedDexopts) {
    return normalizeDexopts(
        Iterables.filter(
            tokenizedDexopts,
            // dexopts have to match exactly since aspect only creates archives for listed ones
            Predicates.in(getAndroidConfig(ruleContext).getDexoptsSupportedInIncrementalDexing())));
  }

  /**
   * Returns the subset of the given dexopts that are forbidden from using incremental dexing by
   * default.
   */
  static Iterable<String> forbiddenDexopts(RuleContext ruleContext, List<String> dexopts) {
    return Iterables.filter(
        dexopts,
        new FlagMatcher(
            getAndroidConfig(ruleContext).getTargetDexoptsThatPreventIncrementalDexing()));
  }

  /**
   * Derives options to use in DexBuilder actions from the given context and dx flags, where the
   * latter typically come from a {@code dexopts} attribute on a top-level target. This should be a
   * superset of {@link #incrementalDexopts}.
   */
  static ImmutableSet<String> topLevelDexbuilderDexopts(Iterable<String> tokenizedDexopts) {
    // We don't need an ordered set but might as well.
    return normalizeDexopts(Iterables.filter(tokenizedDexopts, DEXOPTS_SUPPORTED_IN_DEXBUILDER));
  }

  /**
   * Derives options to use in DexFileMerger actions from the given context and dx flags, where the
   * latter typically come from a {@code dexopts} attribute on a top-level target.
   */
  static ImmutableSet<String> mergerDexopts(
      RuleContext ruleContext, Iterable<String> tokenizedDexopts) {
    // We don't need an ordered set but might as well.  Note we don't need to worry about coverage
    // builds since the merger doesn't use --no-locals.
    return normalizeDexopts(
        Iterables.filter(
            tokenizedDexopts,
            new FlagMatcher(getAndroidConfig(ruleContext).getDexoptsSupportedInDexMerger())));
  }

  /**
   * Derives options to use in DexFileSharder actions from the given context and dx flags, where the
   * latter typically come from a {@code dexopts} attribute on a top-level target.
   */
  static ImmutableSet<String> sharderDexopts(
      RuleContext ruleContext, Iterable<String> tokenizedDexopts) {
    // We don't need an ordered set but might as well.  Note we don't need to worry about coverage
    // builds since the merger doesn't use --no-locals.
    return normalizeDexopts(
        Iterables.filter(
            tokenizedDexopts,
            new FlagMatcher(getAndroidConfig(ruleContext).getDexoptsSupportedInDexSharder())));
  }

  private static ImmutableSet<String> normalizeDexopts(Iterable<String> tokenizedDexopts) {
    // Sort and use ImmutableSet to drop duplicates and get fixed (sorted) order.  Fixed order is
    // important so we generate one dex archive per set of flag in create() method, regardless of
    // how those flags are listed in all the top-level targets being built.
    return Streams.stream(tokenizedDexopts)
        .map(FlagConverter.DX_TO_DEXBUILDER)
        .sorted()
        .collect(ImmutableSet.toImmutableSet()); // collector with dedupe
  }

  private static class FlagMatcher implements Predicate<String> {
    private final ImmutableList<String> matching;

    FlagMatcher(ImmutableList<String> matching) {
      this.matching = matching;
    }

    @Override
    public boolean apply(String input) {
      for (String match : matching) {
        if (input.contains(match)) {
          return true;
        }
      }
      return false;
    }
  }

  private enum FlagConverter implements Function<String, String> {
    DX_TO_DEXBUILDER;

    @Override
    public String apply(String input) {
      return input.replace("--no-", "--no");
    }
  }
}
