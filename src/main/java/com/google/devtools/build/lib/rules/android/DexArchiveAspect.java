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

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.rules.android.AndroidCommon.getAndroidConfig;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import javax.annotation.Nullable;

/**
 * Aspect to {@link DexArchiveProvider build .dex Archives} from Jars.
 */
public final class DexArchiveAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "DexArchiveAspect";
  /**
   * Function that returns a {@link Rule}'s {@code dexopts} attribute for use by this aspect.
   * Must be provided when attaching this aspect to a rule.
   */
  static final Function<Rule, AspectParameters> PARAM_EXTRACTOR =
      new Function<Rule, AspectParameters>() {
        @Nullable
        @Override
        public AspectParameters apply(Rule rule) {
          AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
          AspectParameters.Builder result = new AspectParameters.Builder();
          for (String opt : attributes.get("dexopts", STRING_LIST)) {
            result.addAttribute("dexopts", opt);
          }
          return result.build();
        }
      };

  private static final String ASPECT_DEXBUILDER_PREREQ = "$dex_archive_dexbuilder";
  private static final ImmutableList<String> TRANSITIVE_ATTRIBUTES =
      ImmutableList.of("deps", "exports", "runtime_deps");
  private final String toolsRepository;

  public DexArchiveAspect(String toolsRepository) {
    this.toolsRepository = toolsRepository;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters params) {
    AspectDefinition.Builder result = new AspectDefinition.Builder(NAME)
        // Actually we care about JavaRuntimeJarProvider, but rules don't advertise that provider.
        .requireProvider(JavaCompilationArgsProvider.class)
        .add(attr(ASPECT_DEXBUILDER_PREREQ, LABEL).cfg(HOST).exec()
        // Parse label here since we don't have RuleDefinitionEnvironment.getLabel like in a rule
            .value(Label.parseAbsoluteUnchecked(toolsRepository + "//tools/android:dexbuilder")))
        .requiresConfigurationFragments(AndroidConfiguration.class);
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      result.attributeAspect(attr, this);
    }
    return result.build();
  }

  @Override
  public ConfiguredAspect create(ConfiguredTarget base, RuleContext ruleContext,
      AspectParameters params) throws InterruptedException {
    checkState(base.getProvider(DexArchiveProvider.class) == null,
        "dex archive natively generated: %s", ruleContext.getLabel());

    if (JavaCommon.isNeverLink(ruleContext)) {
      return new ConfiguredAspect.Builder(NAME, ruleContext)
          .addProvider(DexArchiveProvider.class, DexArchiveProvider.NEVERLINK)
          .build();
    }

    DexArchiveProvider.Builder result = createArchiveProviderBuilderFromDeps(ruleContext);
    JavaRuntimeJarProvider jarProvider = base.getProvider(JavaRuntimeJarProvider.class);
    if (jarProvider != null) {
      Collection<String> incrementalDexopts = aspectDexopts(ruleContext, params);
      for (Artifact jar : jarProvider.getRuntimeJars()) {
        Artifact dexArchive = createDexArchiveAction(ruleContext, jar, incrementalDexopts);
        result.addDexArchive(dexArchive, jar);
      }
    }
    return new ConfiguredAspect.Builder(NAME, ruleContext)
        .addProvider(DexArchiveProvider.class, result.build())
        .build();
  }

  private static DexArchiveProvider.Builder createArchiveProviderBuilderFromDeps(
      RuleContext ruleContext) {
    DexArchiveProvider.Builder result = new DexArchiveProvider.Builder();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (ruleContext.getRule().getRuleClassObject().hasAttr(attr, LABEL_LIST)) {
        result.addTransitiveProviders(
            ruleContext.getPrerequisites(attr, Mode.TARGET, DexArchiveProvider.class));
      }
    }
    return result;
  }

  private static Artifact createDexArchiveAction(RuleContext ruleContext, Artifact jar,
      Collection<String> incrementalDexopts) {
    // When DexArchiveAspect is applied to the dependencies of multiple android_binary rules,
    // multiple aspects could end up being applied to shared dependencies. This is a problem because
    // the dexopts from the different android_binary rules could be different, leading to different
    // artifacts. So, the artifacts must be uniquely named. Here, it is convenient to distinguish
    // them by the flags that were used for creating the artifacts.
    String filename = jar.getFilename() + Joiner.on("").join(incrementalDexopts) + ".dex.zip";
    Artifact result = AndroidBinary.getDxArtifact(ruleContext, filename);
    // Aspect must use attribute name for dexbuilder prereq that's different from the prerequisite
    // declared on AndroidBinaryBaseRule because the two prereq's can otherwise name-clash when
    // android_binary targets are built as part of an android_test: building android_test causes
    // the aspect to apply to the android_binary target, but android_binary itself also declares
    // a $dexbuilder prerequisite, so if the aspect also used $dexbuilder then
    // RuleContext.getExecutablePrerequisite would fail with "$dexbuilder produces multiple prereqs"
    // (note they both resolve to the same artifact but that doesn't seem to prevent the exception
    // from being thrown).
    createDexArchiveAction(ruleContext, ASPECT_DEXBUILDER_PREREQ, jar, result, incrementalDexopts);
    return result;
  }

  /**
   * Creates a dex archive using an executable prerequisite called {@code "$dexbuilder"}.  Rules
   * calling this method must declare the appropriate prerequisite, similar to how
   * {@link #getDefinition} does it for {@link DexArchiveAspect} under a different name.
   */
  // Package-private method for use in AndroidBinary
  static void createDexArchiveAction(RuleContext ruleContext, Artifact jar, Artifact dexArchive,
      Collection<String> tokenizedDexopts) {
    createDexArchiveAction(ruleContext, "$dexbuilder", jar, dexArchive,
        incrementalDexingOptions(ruleContext, tokenizedDexopts));
  }

  /**
   * Creates a dexbuilder action with the given input, output, and flags.  Flags must have been
   * filtered and normalized to a set that the dexbuilder tool can understand.
   */
  private static void createDexArchiveAction(RuleContext ruleContext, String dexbuilderPrereq,
      Artifact jar, Artifact dexArchive, Collection<String> incrementalDexopts) {
    // Write command line arguments into a params file for compatibility with WorkerSpawnStrategy
    CustomCommandLine.Builder args = new CustomCommandLine.Builder()
        .addExecPath("--input_jar", jar)
        .addExecPath("--output_zip", dexArchive)
        .add(incrementalDexopts);
    Artifact paramFile =
        ruleContext.getDerivedArtifact(
            ParameterFile.derivePath(dexArchive.getRootRelativePath()), dexArchive.getRoot());
    ruleContext.registerAction(
        new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            paramFile,
            args.build(),
            ParameterFile.ParameterFileType.UNQUOTED,
            ISO_8859_1));
    SpawnAction.Builder dexbuilder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite(dexbuilderPrereq, Mode.HOST))
        // WorkerSpawnStrategy expects the last argument to be @paramfile
        .addArgument("@" + paramFile.getExecPathString())
        .addInput(jar)
        .addInput(paramFile)
        .addOutput(dexArchive)
        .setMnemonic("DexBuilder")
        .setProgressMessage(
            "Dexing " + jar.prettyPrint() + " with applicable dexopts " + incrementalDexopts);
    ruleContext.registerAction(dexbuilder.build(ruleContext));
  }

  private static List<String> aspectDexopts(RuleContext ruleContext, AspectParameters params) {
    return incrementalDexingOptions(ruleContext,
        // Extract dexopts attribute similar to how AndroidBinary does it
        // (minus $(location) expansion)
        tokenizeAndExpandMakeVars(ruleContext, "dexopts", params.getAttribute("dexopts")));
  }

  /**
   * Derives options to use in incremental dexing actions from the given context and dx flags, where
   * the latter typically come from a {@code dexopts} attribute on a top-level target.  This method
   * only works reliably if the given dexopts were tokenized, e.g., using
   * {@link RuleContext#getTokenizedStringListAttr}.
   */
  static List<String> incrementalDexingOptions(RuleContext ruleContext, Iterable<String> dexopts) {
    // Use TreeSet to drop duplicates and get fixed (sorted) order.  Fixed order is important so
    // we generate one dex archive per set of flag in create() method, regardless of how those flags
    // are listed in all the top-level targets being built.
    Set<String> args = new TreeSet<>();
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      // Match what we do in AndroidCommon.createDexAction
      args.add("--nolocals"); // TODO(bazel-team): Still needed? See createDexAction
    }
    Iterables.addAll(args,
        Iterables.transform(
            Iterables.filter(
                dexopts,
                new FlagMatcher(
                    getAndroidConfig(ruleContext).getDexoptsSupportedInIncrementalDexing())),
            FlagConverter.DX_TO_DEXBUILDER));
    return ImmutableList.copyOf(args);
  }

  private static Collection<String> tokenizeAndExpandMakeVars(RuleContext ruleContext,
      String attributeName, Collection<String> original) {
    List<String> result = new ArrayList<>();
    for (String value : original) {
      ruleContext.tokenizeAndExpandMakeVars(result, attributeName, value);
    }
    return result;
  }

  private enum FlagConverter implements Function<String, String> {
    DX_TO_DEXBUILDER;

    @Override
    public String apply(String input) {
      return input.replace("--no-", "--no");
    }
  }

  static class FlagMatcher implements Predicate<String> {

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
}
