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
import static com.google.devtools.build.lib.packages.SkylarkProviderIdentifier.forKey;
import static com.google.devtools.build.lib.rules.android.AndroidCommon.getAndroidConfig;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.rules.java.proto.JavaProtoAspectCommon;
import com.google.devtools.build.lib.rules.java.proto.JavaProtoLibraryAspectProvider;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Aspect to {@link DexArchiveProvider build .dex Archives} from Jars. */
public final class DexArchiveAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "DexArchiveAspect";

  /**
   * Function that returns a {@link Rule}'s {@code incremental_dexing} attribute for use by this
   * aspect. Must be provided when attaching this aspect to a target.
   */
  @AutoCodec
  public static final Function<Rule, AspectParameters> PARAM_EXTRACTOR =
      (Rule rule) -> {
        AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
        AspectParameters.Builder result = new AspectParameters.Builder();
        TriState incrementalAttr = attributes.get("incremental_dexing", TRISTATE);
        result.addAttribute("incremental_dexing", incrementalAttr.name());
        return result.build();
      };
  /**
   * Function that limits this aspect to Java 8 desugaring (disabling incremental dexing) when
   * attaching this aspect to a target. This is intended for implicit attributes like the stub APKs
   * for {@code bazel mobile-install}.
   */
  @AutoCodec
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
          ":android_sdk",
          "aidl_lib", // for the aidl runtime in the android_sdk rule
          "$toolchain", // this is _toolchain in Skylark rules (b/78647825)
          // To get from proto_library through proto_lang_toolchain rule to proto runtime library.
          JavaProtoAspectCommon.LITE_PROTO_TOOLCHAIN_ATTR, "runtime");

  private static final FlagMatcher DEXOPTS_SUPPORTED_IN_DEXBUILDER =
      new FlagMatcher(
          ImmutableList.of("--no-locals", "--no-optimize", "--no-warnings", "--positions"));

  private final String toolsRepository;

  public DexArchiveAspect(String toolsRepository) {
    this.toolsRepository = toolsRepository;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters params) {
    AspectDefinition.Builder result =
        new AspectDefinition.Builder(this)
            .requireSkylarkProviders(forKey(JavaInfo.PROVIDER.getKey()))
            // Latch onto Skylark toolchains in case they have a "runtime" (b/78647825)
            .requireSkylarkProviders(forKey(ToolchainInfo.PROVIDER.getKey()))
            // For android_sdk rules, where we just want to get at aidl runtime deps.
            .requireSkylarkProviders(forKey(AndroidSdkProvider.PROVIDER.getKey()))
            .requireSkylarkProviders(forKey(ProtoInfo.PROVIDER.getKey()))
            .requireProviderSets(
                ImmutableList.of(
                    // For proto_lang_toolchain rules, where we just want to get at their runtime
                    // deps.
                    ImmutableSet.<Class<?>>of(ProtoLangToolchainProvider.class)))
            // Parse labels since we don't have RuleDefinitionEnvironment.getLabel like in a rule
            .add(
                attr(ASPECT_DESUGAR_PREREQ, LABEL)
                    .cfg(HostTransition.createFactory())
                    .exec()
                    .value(
                        Label.parseAbsoluteUnchecked(
                            toolsRepository + "//tools/android:desugar_java8")))
            // Access to --android_sdk so we can stub in a bootclasspath for desugaring if missing
            .add(
                attr(":dex_archive_android_sdk", LABEL)
                    .allowedRuleClasses("android_sdk", "filegroup")
                    .value(
                        AndroidRuleClasses.getAndroidSdkLabel(
                            Label.parseAbsoluteUnchecked(
                                toolsRepository + AndroidRuleClasses.DEFAULT_SDK))))
            .requiresConfigurationFragments(AndroidConfiguration.class)
            .requireAspectsWithProviders(
                ImmutableList.of(ImmutableSet.of(forKey(JavaInfo.PROVIDER.getKey()))))
            .requireAspectsWithNativeProviders(JavaProtoLibraryAspectProvider.class);
    if (TriState.valueOf(params.getOnlyValueOfAttribute("incremental_dexing")) != TriState.NO) {
      // Marginally improves "query2" precision for targets that disable incremental dexing
      result.add(
          attr(ASPECT_DEXBUILDER_PREREQ, LABEL)
              .cfg(HostTransition.createFactory())
              .exec()
              .value(Label.parseAbsoluteUnchecked(toolsRepository + "//tools/android:dexbuilder")));
    }
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      result.propagateAlongAttribute(attr);
    }
    return result.build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTargetAndData ctadBase,
      RuleContext ruleContext,
      AspectParameters params,
      String toolsRepository)
      throws InterruptedException, ActionConflictException {
    ConfiguredAspect.Builder result = new ConfiguredAspect.Builder(this, params, ruleContext);
    Function<Artifact, Artifact> desugaredJars =
        desugarJarsIfRequested(ctadBase.getConfiguredTarget(), ruleContext, result);

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

    DexArchiveProvider.Builder dexArchives =
        new DexArchiveProvider.Builder()
            .addTransitiveProviders(collectPrerequisites(ruleContext, DexArchiveProvider.class));
    Iterable<Artifact> runtimeJars =
        getProducedRuntimeJars(ctadBase.getConfiguredTarget(), ruleContext);
    if (runtimeJars != null) {
      boolean basenameClash = checkBasenameClash(runtimeJars);
      Set<Set<String>> aspectDexopts = aspectDexopts(ruleContext);
      for (Artifact jar : runtimeJars) {
        for (Set<String> incrementalDexopts : aspectDexopts) {
          // Since we're potentially dexing the same jar multiple times with different flags, we
          // need to write unique artifacts for each flag combination. Here, it is convenient to
          // distinguish them by putting the flags that were used for creating the artifacts into
          // their filenames.
          String uniqueFilename =
              (basenameClash ? jar.getRootRelativePathString() : jar.getFilename())
                  + Joiner.on("").join(incrementalDexopts)
                  + ".dex.zip";
          Artifact dexArchive =
              createDexArchiveAction(
                  ruleContext,
                  ASPECT_DEXBUILDER_PREREQ,
                  desugaredJars.apply(jar),
                  incrementalDexopts,
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
      ConfiguredTarget base, RuleContext ruleContext, ConfiguredAspect.Builder result) {
    if (!getAndroidConfig(ruleContext).desugarJava8()) {
      return Functions.identity();
    }
    Map<Artifact, Artifact> newlyDesugared = new HashMap<>();
    if (JavaCommon.isNeverLink(ruleContext)) {
      result.addProvider(AndroidRuntimeJarProvider.NEVERLINK);
      return Functions.forMap(newlyDesugared);
    }
    AndroidRuntimeJarProvider.Builder desugaredJars =
        new AndroidRuntimeJarProvider.Builder()
            .addTransitiveProviders(
                collectPrerequisites(ruleContext, AndroidRuntimeJarProvider.class));
    if (isProtoLibrary(ruleContext)) {
      // TODO(b/33557068): Desugar protos if needed instead of assuming they don't need desugaring
      result.addProvider(desugaredJars.build());
      return Functions.identity();
    }

    JavaInfo javaInfo = JavaInfo.getJavaInfo(base);
    if (javaInfo != null) {
      // These are all transitive hjars of dependencies and hjar of the jar itself
      NestedSet<Artifact> compileTimeClasspath =
          getJavaCompilationArgsProvider(base, ruleContext).getTransitiveCompileTimeJars();
      ImmutableSet.Builder<Artifact> jars = ImmutableSet.builder();
      jars.addAll(javaInfo.getDirectRuntimeJars());

      // If the target is an android_library, it may be a Starlark android_library in which case
      // get the R.jar from the AndroidIdeInfoProvider.
      if (isAndroidLibrary(ruleContext)) {
        Artifact rJar = getAndroidLibraryRJar(base);
        if (rJar != null) {
          // TODO(b/124540821): Disable R.jar desugaring (with a flag).
          jars.add(rJar);
        }
      }

      // For android_* targets we need to honor their bootclasspath (nicer in general to do so)
      NestedSet<Artifact> bootclasspath = getBootclasspath(base, ruleContext);

      ImmutableSet<Artifact> jarsToProcess = jars.build();
      boolean basenameClash = checkBasenameClash(jarsToProcess);
      for (Artifact jar : jarsToProcess) {
        Artifact desugared =
            createDesugarAction(
                ruleContext, basenameClash, jar, bootclasspath, compileTimeClasspath);
        newlyDesugared.put(jar, desugared);
        desugaredJars.addDesugaredJar(jar, desugared);
      }
    }
    result.addProvider(desugaredJars.build());
    return Functions.forMap(newlyDesugared);
  }

  private static Iterable<Artifact> getProducedRuntimeJars(
      ConfiguredTarget base, RuleContext ruleContext) {
    if (isProtoLibrary(ruleContext)) {
      if (!ruleContext.getPrerequisites("srcs", Mode.TARGET).isEmpty()) {
        JavaRuleOutputJarsProvider outputJarsProvider =
            base.getProvider(JavaRuleOutputJarsProvider.class);
        if (outputJarsProvider != null) {
          return outputJarsProvider
              .getOutputJars()
              .stream()
              .map(OutputJar::getClassJar)
              .collect(toImmutableList());
        } else {
          JavaInfo javaInfo = JavaInfo.getJavaInfo(base);
          if (javaInfo != null) {
            return javaInfo.getDirectRuntimeJars();
          }
        }
      }
    } else {
      ImmutableSet.Builder<Artifact> jars = ImmutableSet.builder();
      JavaInfo javaInfo = JavaInfo.getJavaInfo(base);
      if (javaInfo != null) {
        jars.addAll(javaInfo.getDirectRuntimeJars());
      }

      // The Starlark android_library does not put the R.jar file in the direct runtime deps of the
      // JavaInfo, it can only be retrieved from the AndroidIdeInfoProvider.
      if (isAndroidLibrary(ruleContext)) {
        Artifact rJar = getAndroidLibraryRJar(base);
        if (rJar != null) {
          jars.add(rJar);
        }
      }

      return jars.build();
    }
    return null;
  }

  private static JavaCompilationArgsProvider getJavaCompilationArgsProvider(
      ConfiguredTarget base, RuleContext ruleContext) {
    JavaCompilationArgsProvider provider =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, base);
    if (provider != null) {
      return provider;
    }
    return isProtoLibrary(ruleContext) ? base.getProvider(JavaCompilationArgsProvider.class) : null;
  }

  private static boolean isAndroidLibrary(RuleContext ruleContext) {
    return "android_library".equals(ruleContext.getRule().getRuleClass());
  }

  private static boolean isProtoLibrary(RuleContext ruleContext) {
    return "proto_library".equals(ruleContext.getRule().getRuleClass());
  }

  private static Artifact getAndroidLibraryRJar(ConfiguredTarget base) {
    AndroidIdeInfoProvider provider =
        (AndroidIdeInfoProvider) base.get(AndroidIdeInfoProvider.PROVIDER.getKey());
    if (provider != null && provider.getResourceJar() != null) {
      return provider.getResourceJar().getClassJar();
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

  private static <T extends TransitiveInfoProvider> Iterable<T> collectPrerequisites(
      RuleContext ruleContext, Class<T> classType) {
    IterablesChain.Builder<T> result = IterablesChain.builder();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (ruleContext.attributes().getAttributeType(attr) != null) {
        result.add(ruleContext.getPrerequisites(attr, Mode.TARGET, classType));
      }
    }
    return result.build();
  }

  private static NestedSet<Artifact> getBootclasspath(
      ConfiguredTarget base, RuleContext ruleContext) {
    JavaCompilationInfoProvider compilationInfo =
        JavaInfo.getProvider(JavaCompilationInfoProvider.class, base);
    if (compilationInfo == null || compilationInfo.getBootClasspath().isEmpty()) {
      return NestedSetBuilder.<Artifact>naiveLinkOrder()
          .add(
              ruleContext
                  .getPrerequisite(
                      ":dex_archive_android_sdk", Mode.TARGET, AndroidSdkProvider.PROVIDER)
                  .getAndroidJar())
          .build();
    }
    return compilationInfo.getBootClasspathAsNestedSet();
  }

  private Artifact createDesugarAction(
      RuleContext ruleContext,
      boolean disambiguateBasenames,
      Artifact jar,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> compileTimeClasspath) {
    return createDesugarAction(
        ruleContext,
        ASPECT_DESUGAR_PREREQ,
        jar,
        bootclasspath,
        compileTimeClasspath,
        AndroidBinary.getDxArtifact(
            ruleContext,
            (disambiguateBasenames ? jar.getRootRelativePathString() : jar.getFilename())
                + "_desugared.jar"));
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
      Artifact result) {
    return createDesugarAction(ruleContext, "$desugar", jar, bootclasspath, classpath, result);
  }

  private static Artifact createDesugarAction(
      RuleContext ruleContext,
      String desugarPrereqName,
      Artifact jar,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> classpath,
      Artifact result) {
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

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite(desugarPrereqName, Mode.HOST))
            .addInput(jar)
            .addInputs(bootclasspath)
            .addTransitiveInputs(classpath)
            .addOutput(result)
            .setMnemonic("Desugar")
            .setProgressMessage("Desugaring %s for Android", jar.prettyPrint())
            .setExecutionInfo(ExecutionRequirements.WORKER_MODE_ENABLED)
            .addCommandLine(
                // Always use params file, so we don't need to compute command line length first
                args.build(), ParamFileInfo.builder(UNQUOTED).setUseAlways(true).build())
            .build(ruleContext));
    return result;
  }

  /**
   * Creates a dexbuilder action with the given input, output, and flags. Flags must have been
   * filtered and normalized to a set that the dexbuilder tool can understand.
   *
   * @return the artifact given as {@code result}, which can simplify calling code
   */
  // Package-private method for use in AndroidBinary
  static Artifact createDexArchiveAction(
      RuleContext ruleContext,
      String dexbuilderPrereq,
      Artifact jar,
      Set<String> incrementalDexopts,
      Artifact dexArchive) {
    CustomCommandLine args =
        new CustomCommandLine.Builder()
            .addExecPath("--input_jar", jar)
            .addExecPath("--output_zip", dexArchive)
            .addAll(ImmutableList.copyOf(incrementalDexopts))
            .build();
    SpawnAction.Builder dexbuilder =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite(dexbuilderPrereq, Mode.HOST))
            // WorkerSpawnStrategy expects the last argument to be @paramfile
            .addInput(jar)
            .addOutput(dexArchive)
            .setMnemonic("DexBuilder")
            .setProgressMessage(
                "Dexing %s with applicable dexopts %s", jar.prettyPrint(), incrementalDexopts)
            // Always use params file for compatibility with WorkerSpawnStrategy
            .addCommandLine(args, ParamFileInfo.builder(UNQUOTED).setUseAlways(true).build());
    if (getAndroidConfig(ruleContext).useWorkersWithDexbuilder()) {
      dexbuilder.setExecutionInfo(ExecutionRequirements.WORKER_MODE_ENABLED);
    }
    ruleContext.registerAction(dexbuilder.build(ruleContext));
    return dexArchive;
  }

  private static Set<Set<String>> aspectDexopts(RuleContext ruleContext) {
    return Sets.powerSet(
        normalizeDexopts(getAndroidConfig(ruleContext).getDexoptsSupportedInIncrementalDexing()));
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
   * Returns the subset of the given dexopts that are blacklisted from using incremental dexing by
   * default.
   */
  static Iterable<String> blacklistedDexopts(RuleContext ruleContext, List<String> dexopts) {
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
