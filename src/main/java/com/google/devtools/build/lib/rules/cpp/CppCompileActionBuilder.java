// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.DotdFile;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.IncludeResolver;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * Builder class to construct C++ compile actions.
 */
public class CppCompileActionBuilder {
  public static final UUID GUID = UUID.fromString("cee5db0a-d2ad-4c69-9b81-97c936a29075");

  private final ActionOwner owner;
  private final List<String> features = new ArrayList<>();
  private CcToolchainFeatures.FeatureConfiguration featureConfiguration;
  private CcToolchainFeatures.Variables variables;
  private final Artifact sourceFile;
  private final Label sourceLabel;
  private final NestedSetBuilder<Artifact> mandatoryInputsBuilder;
  private NestedSetBuilder<Artifact> pluginInputsBuilder;
  private Artifact optionalSourceFile;
  private Artifact outputFile;
  private PathFragment tempOutputFile;
  private DotdFile dotdFile;
  private Artifact gcnoFile;
  private final BuildConfiguration configuration;
  private CppCompilationContext context = CppCompilationContext.EMPTY;
  private final List<String> copts = new ArrayList<>();
  private final List<String> pluginOpts = new ArrayList<>();
  private final List<Pattern> nocopts = new ArrayList<>();
  private AnalysisEnvironment analysisEnvironment;
  private ImmutableList<PathFragment> extraSystemIncludePrefixes = ImmutableList.of();
  private String fdoBuildStamp;
  private boolean usePic; 
  private IncludeResolver includeResolver = CppCompileAction.VOID_INCLUDE_RESOLVER;
  private UUID actionClassId = GUID;
  private Class<? extends CppCompileActionContext> actionContext;
  private CppConfiguration cppConfiguration;
  private ImmutableMap<Artifact, IncludeScannable> lipoScannableMap;
  private RuleContext ruleContext = null;
  private Boolean shouldScanIncludes;
  // New fields need to be added to the copy constructor.

  /**
   * Creates a builder from a rule. This also uses the configuration and
   * artifact factory from the rule.
   */
  public CppCompileActionBuilder(RuleContext ruleContext, Artifact sourceFile, Label sourceLabel) {
    this.owner = ruleContext.getActionOwner();
    this.actionContext = CppCompileActionContext.class;
    this.cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    this.analysisEnvironment = ruleContext.getAnalysisEnvironment();
    this.sourceFile = sourceFile;
    this.sourceLabel = sourceLabel;
    this.configuration = ruleContext.getConfiguration();
    this.mandatoryInputsBuilder = NestedSetBuilder.stableOrder();
    this.pluginInputsBuilder = NestedSetBuilder.stableOrder();
    this.lipoScannableMap = getLipoScannableMap(ruleContext);
    this.ruleContext = ruleContext;

    features.addAll(ruleContext.getFeatures());
  }

  private static ImmutableMap<Artifact, IncludeScannable> getLipoScannableMap(
      RuleContext ruleContext) {
    if (!ruleContext.getFragment(CppConfiguration.class).isLipoOptimization()
        // Rules that do not contain sources that are compiled into object files, but may
        // contain headers, will still create CppCompileActions without providing a
        // lipo_context_collector.
        || ruleContext.attributes().getAttributeDefinition(":lipo_context_collector") == null) {
      return null;
    }
    LipoContextProvider provider = ruleContext.getPrerequisite(
        ":lipo_context_collector", Mode.DONT_CHECK, LipoContextProvider.class);
    return provider.getIncludeScannables();
  }

  /**
   * Creates a builder for an owner that is not required to be rule.
   * 
   * <p>If errors are found when creating the {@code CppCompileAction}, builders constructed
   * this way will throw a runtime exception.
   */
  @VisibleForTesting
  public CppCompileActionBuilder(
      ActionOwner owner, AnalysisEnvironment analysisEnvironment, Artifact sourceFile,
      Label sourceLabel, BuildConfiguration configuration) {
    this.owner = owner;
    this.actionContext = CppCompileActionContext.class;
    this.cppConfiguration = configuration.getFragment(CppConfiguration.class);
    this.analysisEnvironment = analysisEnvironment;
    this.sourceFile = sourceFile;
    this.sourceLabel = sourceLabel;
    this.configuration = configuration;
    this.mandatoryInputsBuilder = NestedSetBuilder.stableOrder();
    this.pluginInputsBuilder = NestedSetBuilder.stableOrder();
    this.lipoScannableMap = ImmutableMap.of();
  }

  /**
   * Creates a builder that is a copy of another builder.
   */
  public CppCompileActionBuilder(CppCompileActionBuilder other) {
    this.owner = other.owner;
    this.features.addAll(other.features);
    this.featureConfiguration = other.featureConfiguration;
    this.sourceFile = other.sourceFile;
    this.sourceLabel = other.sourceLabel;
    this.mandatoryInputsBuilder = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(other.mandatoryInputsBuilder.build());
    this.pluginInputsBuilder = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(other.pluginInputsBuilder.build());
    this.optionalSourceFile = other.optionalSourceFile;
    this.outputFile = other.outputFile;
    this.tempOutputFile = other.tempOutputFile;
    this.dotdFile = other.dotdFile;
    this.gcnoFile = other.gcnoFile;
    this.configuration = other.configuration;
    this.context = other.context;
    this.copts.addAll(other.copts);
    this.pluginOpts.addAll(other.pluginOpts);
    this.nocopts.addAll(other.nocopts);
    this.analysisEnvironment = other.analysisEnvironment;
    this.extraSystemIncludePrefixes = ImmutableList.copyOf(other.extraSystemIncludePrefixes);
    this.includeResolver = other.includeResolver;
    this.actionClassId = other.actionClassId;
    this.actionContext = other.actionContext;
    this.cppConfiguration = other.cppConfiguration;
    this.fdoBuildStamp = other.fdoBuildStamp;
    this.usePic = other.usePic;
    this.lipoScannableMap = other.lipoScannableMap;
    this.ruleContext = other.ruleContext;
    this.shouldScanIncludes = other.shouldScanIncludes;
  }

  public PathFragment getTempOutputFile() {
    return tempOutputFile;
  }

  public Artifact getSourceFile() {
    return sourceFile;
  }

  public CppCompilationContext getContext() {
    return context;
  }

  public NestedSet<Artifact> getMandatoryInputs() {
    return mandatoryInputsBuilder.build();
  }

  /**
   * Returns the .dwo output file that matches the specified .o output file. If Fission mode
   * isn't enabled for this build, this is null (we don't produce .dwo files in that case).
   */
  private static Artifact getDwoFile(RuleContext ruleContext, Artifact outputFile,
      CppConfiguration cppConfiguration) {

    // Only create .dwo's for .o compilations (i.e. not .ii or .S).
    boolean isObjectOutput = CppFileTypes.OBJECT_FILE.matches(outputFile.getExecPath())
        || CppFileTypes.PIC_OBJECT_FILE.matches(outputFile.getExecPath());

    // Note configurations can be null for tests.
    if (cppConfiguration != null && cppConfiguration.useFission() && isObjectOutput) {
      return ruleContext.getRelatedArtifact(outputFile.getRootRelativePath(), ".dwo");
    } else {
      return null;
    }
  }

  private static Predicate<String> getNocoptPredicate(Collection<Pattern> patterns) {
    final ImmutableList<Pattern> finalPatterns = ImmutableList.copyOf(patterns);
    if (finalPatterns.isEmpty()) {
      return Predicates.alwaysTrue();
    } else {
      return new Predicate<String>() {
        @Override
        public boolean apply(String option) {
          for (Pattern pattern : finalPatterns) {
            if (pattern.matcher(option).matches()) {
              return false;
            }
          }

          return true;
        }
      };
    }
  }

  private Iterable<IncludeScannable> getLipoScannables(NestedSet<Artifact> realMandatoryInputs) {
    return lipoScannableMap == null ? ImmutableList.<IncludeScannable>of() : Iterables.filter(
        Iterables.transform(
            Iterables.filter(
                FileType.filter(
                    realMandatoryInputs,
                    CppFileTypes.C_SOURCE, CppFileTypes.CPP_SOURCE,
                    CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR),
                Predicates.not(Predicates.equalTo(getSourceFile()))),
            Functions.forMap(lipoScannableMap, null)),
        Predicates.notNull());
  }

  /**
   * Builds the Action as configured and returns the to be generated Artifact.
   *
   * <p>This method may be called multiple times to create multiple compile
   * actions (usually after calling some setters to modify the generated
   * action).
   */
  public CppCompileAction build() {
    // This must be set either to false or true by CppSemantics, otherwise someone forgot to call
    // finalizeCompileActionBuilder on this builder.
    Preconditions.checkNotNull(shouldScanIncludes);

    // Configuration can be null in tests.
    NestedSetBuilder<Artifact> realMandatoryInputsBuilder = NestedSetBuilder.compileOrder();
    realMandatoryInputsBuilder.addTransitive(mandatoryInputsBuilder.build());
    if (tempOutputFile == null && !shouldScanIncludes) {
      realMandatoryInputsBuilder.addTransitive(context.getDeclaredIncludeSrcs());
    }
    realMandatoryInputsBuilder.addTransitive(context.getAdditionalInputs());
    realMandatoryInputsBuilder.addTransitive(pluginInputsBuilder.build());
    realMandatoryInputsBuilder.add(sourceFile);
    boolean fake = tempOutputFile != null;

    // Copying the collections is needed to make the builder reusable.
    if (fake) {
      return new FakeCppCompileAction(owner, ImmutableList.copyOf(features), featureConfiguration,
          variables, sourceFile, shouldScanIncludes, sourceLabel,
          realMandatoryInputsBuilder.build(), outputFile,
          tempOutputFile, dotdFile, configuration, cppConfiguration, context, actionContext,
          ImmutableList.copyOf(copts), ImmutableList.copyOf(pluginOpts),
          getNocoptPredicate(nocopts), extraSystemIncludePrefixes, fdoBuildStamp, ruleContext,
          usePic);
    } else {
      NestedSet<Artifact> realMandatoryInputs = realMandatoryInputsBuilder.build();

      return new CppCompileAction(owner, ImmutableList.copyOf(features), featureConfiguration,
          variables, sourceFile, shouldScanIncludes, sourceLabel, realMandatoryInputs,
          outputFile, dotdFile, gcnoFile, getDwoFile(ruleContext, outputFile, cppConfiguration),
          optionalSourceFile, configuration, cppConfiguration, context,
          actionContext, ImmutableList.copyOf(copts),
          ImmutableList.copyOf(pluginOpts),
          getNocoptPredicate(nocopts),
          extraSystemIncludePrefixes, fdoBuildStamp,
          includeResolver, getLipoScannables(realMandatoryInputs), actionClassId,
          usePic, ruleContext);
    }
  }
  
  /**
   * Sets the feature configuration to be used for the action. 
   */
  public CppCompileActionBuilder setFeatureConfiguration(
      FeatureConfiguration featureConfiguration) {
    this.featureConfiguration = featureConfiguration;
    return this;
  }
  
  /**
   * Sets the feature build variables to be used for the action. 
   */
  public CppCompileActionBuilder setVariables(CcToolchainFeatures.Variables variables) {
    this.variables = variables;
    return this;
  }

  public CppCompileActionBuilder setIncludeResolver(IncludeResolver includeResolver) {
    this.includeResolver = includeResolver;
    return this;
  }

  public CppCompileActionBuilder setCppConfiguration(CppConfiguration cppConfiguration) {
    this.cppConfiguration = cppConfiguration;
    return this;
  }

  public CppCompileActionBuilder setActionContext(
      Class<? extends CppCompileActionContext> actionContext) {
    this.actionContext = actionContext;
    return this;
  }

  public CppCompileActionBuilder setActionClassId(UUID uuid) {
    this.actionClassId = uuid;
    return this;
  }

  public CppCompileActionBuilder setExtraSystemIncludePrefixes(
      Collection<PathFragment> extraSystemIncludePrefixes) {
    this.extraSystemIncludePrefixes = ImmutableList.copyOf(extraSystemIncludePrefixes);
    return this;
  }

  public CppCompileActionBuilder addPluginInput(Artifact artifact) {
    pluginInputsBuilder.add(artifact);
    return this;
  }

  public CppCompileActionBuilder clearPluginInputs() {
    pluginInputsBuilder = NestedSetBuilder.stableOrder();
    return this;
  }

  /**
   * Set an optional source file (usually with metadata of the main source file). The optional
   * source file can only be set once, whether via this method or through the constructor
   * {@link #CppCompileActionBuilder(CppCompileActionBuilder)}.
   */
  public CppCompileActionBuilder addOptionalSourceFile(Artifact artifact) {
    Preconditions.checkState(optionalSourceFile == null, "%s %s", optionalSourceFile, artifact);
    optionalSourceFile = artifact;
    return this;
  }

  public CppCompileActionBuilder addMandatoryInputs(Iterable<Artifact> artifacts) {
    mandatoryInputsBuilder.addAll(artifacts);
    return this;
  }

  public CppCompileActionBuilder addTransitiveMandatoryInputs(NestedSet<Artifact> artifacts) {
    mandatoryInputsBuilder.addTransitive(artifacts);
    return this;
  }

  public CppCompileActionBuilder setOutputFile(Artifact outputFile) {
    this.outputFile = outputFile;
    return this;
  }

  /**
   * The temp output file is not an artifact, since it does not appear in the outputs of the
   * action.
   *
   * <p>This is theoretically a problem if that file already existed before, since then Blaze
   * does not delete it before executing the rule, but 1. that only applies for local
   * execution which does not happen very often and 2. it is only a problem if the compiler is
   * affected by the presence of this file, which it should not be.
   */
  public CppCompileActionBuilder setTempOutputFile(PathFragment tempOutputFile) {
    this.tempOutputFile = tempOutputFile;
    return this;
  }

  @VisibleForTesting
  public CppCompileActionBuilder setDotdFileForTesting(Artifact dotdFile) {
    this.dotdFile = new DotdFile(dotdFile);
    return this;
  }

  public CppCompileActionBuilder setDotdFile(PathFragment outputName, String extension) {
    if (CppFileTypes.mustProduceDotdFile(outputName.toString())) {
      if (configuration.getFragment(CppConfiguration.class).getInmemoryDotdFiles()) {
        // Just set the path, no artifact is constructed
        PathFragment file = FileSystemUtils.replaceExtension(outputName, extension);
        Root root = configuration.getBinDirectory();
        dotdFile = new DotdFile(root.getExecPath().getRelative(file));
      } else {
        dotdFile = new DotdFile(ruleContext.getRelatedArtifact(outputName, extension));
      }
    } else {
      dotdFile = null;
    }
    return this;
  }

  public CppCompileActionBuilder setGcnoFile(Artifact gcnoFile) {
    this.gcnoFile = gcnoFile;
    return this;
  }

  public CppCompileActionBuilder addCopt(String copt) {
    copts.add(copt);
    return this;
  }

  public CppCompileActionBuilder addPluginOpt(String opt) {
    pluginOpts.add(opt);
    return this;
  }

  public CppCompileActionBuilder clearPluginOpts() {
    pluginOpts.clear();
    return this;
  }

  public CppCompileActionBuilder addCopts(Iterable<? extends String> copts) {
    Iterables.addAll(this.copts, copts);
    return this;
  }

  public CppCompileActionBuilder addCopts(int position, Iterable<? extends String> copts) {
    this.copts.addAll(position, ImmutableList.copyOf(copts));
    return this;
  }

  public CppCompileActionBuilder addNocopts(Pattern nocopts) {
    this.nocopts.add(nocopts);
    return this;
  }

  public CppCompileActionBuilder setContext(CppCompilationContext context) {
    this.context = context;
    return this;
  }

  public CppCompileActionBuilder setFdoBuildStamp(String fdoBuildStamp) {
    this.fdoBuildStamp = fdoBuildStamp;
    return this;
  }

  /**
   * Sets whether the CompileAction should use pic mode.
   */
  public CppCompileActionBuilder setPicMode(boolean usePic) {
    this.usePic = usePic;
    return this;
  }

  public void setShouldScanIncludes(boolean shouldScanIncludes) {
    this.shouldScanIncludes = shouldScanIncludes;
  }

  public boolean getShouldScanIncludes() {
    return shouldScanIncludes;
  }
}
