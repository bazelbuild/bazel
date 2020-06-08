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
package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A helper class for collecting instrumented files and metadata for a target.
 */
public final class InstrumentedFilesCollector {

  private InstrumentedFilesCollector() {}

  /**
   * Forwards any instrumented files from the given target's dependencies (as defined in {@code
   * dependencyAttributes}) for further export. No files from this target are considered
   * instrumented.
   *
   * @return instrumented file provider of all dependencies in {@code dependencyAttributes}
   */
  public static InstrumentedFilesInfo forward(
      RuleContext ruleContext, String... dependencyAttributes) {
    return collect(
        ruleContext,
        new InstrumentationSpec(FileTypeSet.NO_FILE).withDependencyAttributes(dependencyAttributes),
        /* localMetadataCollector= */ null,
        /* rootFiles= */ null,
        /* reportedToActualSources= */ NestedSetBuilder.<Pair<String, String>>emptySet(
            Order.STABLE_ORDER));
  }

  public static InstrumentedFilesInfo forwardAll(RuleContext ruleContext) {
    if (!ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      return InstrumentedFilesInfo.EMPTY;
    }
    InstrumentedFilesInfoBuilder instrumentedFilesInfoBuilder =
        new InstrumentedFilesInfoBuilder(ruleContext);
    for (TransitiveInfoCollection dep : getAllNonToolPrerequisites(ruleContext)) {
      instrumentedFilesInfoBuilder.addFromDependency(dep);
    }
    return instrumentedFilesInfoBuilder.build();
  }

  public static InstrumentedFilesInfo collectTransitive(
      RuleContext ruleContext, InstrumentationSpec spec) {
    return collect(
        ruleContext,
        spec,
        NO_METADATA_COLLECTOR,
        ImmutableList.of(),
        /* reportedToActualSources= */ NestedSetBuilder.<Pair<String, String>>emptySet(
            Order.STABLE_ORDER));
  }

  public static InstrumentedFilesInfo collectTransitive(
      RuleContext ruleContext,
      InstrumentationSpec spec,
      NestedSet<Pair<String, String>> reportedToActualSources) {
    return collect(
        ruleContext,
        spec,
        NO_METADATA_COLLECTOR,
        ImmutableList.of(),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Pair<String, String>>emptySet(Order.STABLE_ORDER),
        false,
        reportedToActualSources);
  }

  public static InstrumentedFilesInfo collect(
      RuleContext ruleContext,
      InstrumentationSpec spec,
      LocalMetadataCollector localMetadataCollector,
      Iterable<Artifact> rootFiles) {
    return collect(
        ruleContext,
        spec,
        localMetadataCollector,
        rootFiles,
        /* reportedToActualSources= */ NestedSetBuilder.<Pair<String, String>>emptySet(
            Order.STABLE_ORDER));
  }

  public static InstrumentedFilesInfo collect(
      RuleContext ruleContext,
      InstrumentationSpec spec,
      LocalMetadataCollector localMetadataCollector,
      Iterable<Artifact> rootFiles,
      NestedSet<Pair<String, String>> reportedToActualSources) {
    return collect(
        ruleContext,
        spec,
        localMetadataCollector,
        rootFiles,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Pair<String, String>>emptySet(Order.STABLE_ORDER),
        false,
        reportedToActualSources);
  }

  /**
   * Collects transitive instrumentation data from dependencies, collects local source files from
   * dependencies, collects local metadata files by traversing the action graph of the current
   * configured target, collect rule-specific instrumentation support file sand creates baseline
   * coverage actions for the transitive closure of source files (if <code>withBaselineCoverage
   * </code> is true).
   */
  public static InstrumentedFilesInfo collect(
      RuleContext ruleContext,
      InstrumentationSpec spec,
      LocalMetadataCollector localMetadataCollector,
      Iterable<Artifact> rootFiles,
      NestedSet<Artifact> coverageSupportFiles,
      NestedSet<Pair<String, String>> coverageEnvironment,
      boolean withBaselineCoverage) {
    return collect(
        ruleContext,
        spec,
        localMetadataCollector,
        rootFiles,
        coverageSupportFiles,
        coverageEnvironment,
        withBaselineCoverage,
        /* reportedToActualSources= */ NestedSetBuilder.<Pair<String, String>>emptySet(
            Order.STABLE_ORDER));
  }

  public static InstrumentedFilesInfo collect(
      RuleContext ruleContext,
      InstrumentationSpec spec,
      @Nullable LocalMetadataCollector localMetadataCollector,
      @Nullable Iterable<Artifact> rootFiles,
      NestedSet<Artifact> coverageSupportFiles,
      NestedSet<Pair<String, String>> coverageEnvironment,
      boolean withBaselineCoverage,
      NestedSet<Pair<String, String>> reportedToActualSources) {
    Preconditions.checkNotNull(ruleContext);
    Preconditions.checkNotNull(spec);

    if (!ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      return InstrumentedFilesInfo.EMPTY;
    }

    InstrumentedFilesInfoBuilder instrumentedFilesInfoBuilder =
        new InstrumentedFilesInfoBuilder(
            ruleContext, coverageSupportFiles, coverageEnvironment, reportedToActualSources);

    // Transitive instrumentation data.
    for (TransitiveInfoCollection dep :
        getPrerequisitesForAttributes(ruleContext, spec.dependencyAttributes)) {
      instrumentedFilesInfoBuilder.addFromDependency(dep);
    }

    // Local sources.
    NestedSet<Artifact> localSources = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    if (shouldIncludeLocalSources(
        ruleContext.getConfiguration(), ruleContext.getLabel(), ruleContext.isTestTarget())) {
      NestedSetBuilder<Artifact> localSourcesBuilder = NestedSetBuilder.stableOrder();
      for (TransitiveInfoCollection dep :
          getPrerequisitesForAttributes(ruleContext, spec.sourceAttributes)) {
        if (!spec.splitLists && dep.get(InstrumentedFilesInfo.STARLARK_CONSTRUCTOR) != null) {
          continue;
        }
        for (Artifact artifact : dep.getProvider(FileProvider.class).getFilesToBuild().toList()) {
          if (artifact.isSourceArtifact() &&
              spec.instrumentedFileTypes.matches(artifact.getFilename())) {
            localSourcesBuilder.add(artifact);
          }
        }
      }
      localSources = localSourcesBuilder.build();
    }
    instrumentedFilesInfoBuilder.addLocalSources(localSources);
    if (withBaselineCoverage) {
      // Also add the local sources to the baseline coverage instrumented sources, if the current
      // rule supports baseline coverage.
      // TODO(ulfjack): Generate a local baseline coverage action, and then merge at the leaves.
      instrumentedFilesInfoBuilder.addBaselineCoverageSources(localSources);
    }

    // Local metadata files.
    if (localMetadataCollector != null) {
      instrumentedFilesInfoBuilder.collectLocalMetadata(localMetadataCollector, rootFiles);
    }

    return instrumentedFilesInfoBuilder.build();
  }

  /**
   * Return whether the sources included by {@code target} (a {@link TransitiveInfoCollection}
   * representing a rule) should be instrumented according the --instrumentation_filter and
   * --instrument_test_targets settings in {@code config}.
   */
  public static boolean shouldIncludeLocalSources(BuildConfiguration config,
      TransitiveInfoCollection target) {
    return shouldIncludeLocalSources(config, target.getLabel(),
        target.getProvider(TestProvider.class) != null);
  }

  /**
   * Return whether the sources of the rule in {@code ruleContext} should be instrumented based on
   * the --instrumentation_filter and --instrument_test_targets config settings.
   */
  public static boolean shouldIncludeLocalSources(
      BuildConfiguration config, Label label, boolean isTest) {
    return ((config.shouldInstrumentTestTargets() || !isTest)
        && config.getInstrumentationFilter().isIncluded(label.toString()));
  }

  /**
   * The set of file types and attributes to visit to collect instrumented files for a certain rule
   * type. The class is intentionally immutable, so that a single instance is sufficient for all
   * rules of the same type (and in some cases all rules of related types, such as all {@code foo_*}
   * rules).
   */
  @Immutable
  public static final class InstrumentationSpec {
    private final FileTypeSet instrumentedFileTypes;

    /** The list of attributes which should be checked for sources. */
    private final Collection<String> sourceAttributes;

    /** The list of attributes from which to collect transitive coverage information. */
    private final Collection<String> dependencyAttributes;

    /** Whether the source and dependency lists are separate. */
    private final boolean splitLists;

    public InstrumentationSpec(FileTypeSet instrumentedFileTypes,
        String... instrumentedAttributes) {
      this(instrumentedFileTypes, ImmutableList.copyOf(instrumentedAttributes));
    }

    public InstrumentationSpec(FileTypeSet instrumentedFileTypes,
        Collection<String> instrumentedAttributes) {
      this(instrumentedFileTypes, instrumentedAttributes, instrumentedAttributes, false);
    }

    private InstrumentationSpec(FileTypeSet instrumentedFileTypes,
        Collection<String> instrumentedSourceAttributes,
        Collection<String> instrumentedDependencyAttributes,
        boolean splitLists) {
      this.instrumentedFileTypes = instrumentedFileTypes;
      this.sourceAttributes = ImmutableList.copyOf(instrumentedSourceAttributes);
      this.dependencyAttributes =
          ImmutableList.copyOf(instrumentedDependencyAttributes);
      this.splitLists = splitLists;
    }

    /**
     * Returns a new instrumentation spec with the given attribute names replacing the ones
     * stored in this object.
     */
    public InstrumentationSpec withAttributes(String... instrumentedAttributes) {
      return new InstrumentationSpec(instrumentedFileTypes, instrumentedAttributes);
    }

    /**
     * Returns a new instrumentation spec with the given attribute names replacing the ones
     * stored in this object.
     */
    public InstrumentationSpec withSourceAttributes(String... instrumentedAttributes) {
      return new InstrumentationSpec(instrumentedFileTypes,
          ImmutableList.copyOf(instrumentedAttributes), dependencyAttributes, true);
    }

    /**
     * Returns a new instrumentation spec with the given attribute names replacing the ones
     * stored in this object.
     */
    public InstrumentationSpec withDependencyAttributes(String... instrumentedAttributes) {
      return new InstrumentationSpec(instrumentedFileTypes,
          sourceAttributes, ImmutableList.copyOf(instrumentedAttributes), true);
    }
  }

  /**
   * The implementation for the local metadata collection. The intention is that implementations
   * recurse over the locally (i.e., for that configured target) created actions and collect
   * metadata files.
   */
  public abstract static class LocalMetadataCollector {
    /**
     * Recursively runs over the local actions and add metadata files to the metadataFilesBuilder.
     */
    public abstract void collectMetadataArtifacts(
        Iterable<Artifact> artifacts, AnalysisEnvironment analysisEnvironment,
        NestedSetBuilder<Artifact> metadataFilesBuilder);

    /**
     * Adds action output of a particular type to metadata files.
     *
     * <p>Only adds the first output that matches the given file type.
     *
     * @param metadataFilesBuilder builder to collect metadata files
     * @param action the action whose outputs to scan
     * @param fileType the filetype of outputs which should be collected
     */
    protected void addOutputs(NestedSetBuilder<Artifact> metadataFilesBuilder,
                              ActionAnalysisMetadata action, FileType fileType) {
      for (Artifact output : action.getOutputs()) {
        if (fileType.matches(output.getFilename())) {
          metadataFilesBuilder.add(output);
          break;
        }
      }
    }
  }

  private static class InstrumentedFilesInfoBuilder {

    RuleContext ruleContext;
    NestedSetBuilder<Artifact> instrumentedFilesBuilder;
    NestedSetBuilder<Artifact> metadataFilesBuilder;
    NestedSetBuilder<Artifact> baselineCoverageInstrumentedFilesBuilder;
    NestedSetBuilder<Artifact> coverageSupportFilesBuilder;
    NestedSetBuilder<Pair<String, String>> coverageEnvironmentBuilder;
    NestedSet<Pair<String, String>> reportedToActualSources;

    InstrumentedFilesInfoBuilder(
        RuleContext ruleContext,
        NestedSet<Artifact> coverageSupportFiles,
        NestedSet<Pair<String, String>> coverageEnvironment,
        NestedSet<Pair<String, String>> reportedToActualSources) {
      this.ruleContext = ruleContext;
      instrumentedFilesBuilder = NestedSetBuilder.stableOrder();
      metadataFilesBuilder = NestedSetBuilder.stableOrder();
      baselineCoverageInstrumentedFilesBuilder = NestedSetBuilder.stableOrder();
      coverageSupportFilesBuilder =
          NestedSetBuilder.<Artifact>stableOrder().addTransitive(coverageSupportFiles);
      coverageEnvironmentBuilder =
          NestedSetBuilder.<Pair<String, String>>compileOrder().addTransitive(coverageEnvironment);
      this.reportedToActualSources = reportedToActualSources;
    }

    InstrumentedFilesInfoBuilder(RuleContext ruleContext) {
      this(
          ruleContext,
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Pair<String, String>>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Pair<String, String>>emptySet(Order.STABLE_ORDER));
    }

    void addFromDependency(TransitiveInfoCollection dep) {
      InstrumentedFilesInfo provider = dep.get(InstrumentedFilesInfo.STARLARK_CONSTRUCTOR);
      if (provider != null) {
        instrumentedFilesBuilder.addTransitive(provider.getInstrumentedFiles());
        metadataFilesBuilder.addTransitive(provider.getInstrumentationMetadataFiles());
        baselineCoverageInstrumentedFilesBuilder.addTransitive(
            provider.getBaselineCoverageInstrumentedFiles());
        coverageSupportFilesBuilder.addTransitive(provider.getCoverageSupportFiles());
        coverageEnvironmentBuilder.addTransitive(provider.getCoverageEnvironment());
      }
    }

    void addLocalSources(NestedSet<Artifact> localSources) {
      instrumentedFilesBuilder.addTransitive(localSources);
    }

    void addBaselineCoverageSources(NestedSet<Artifact> localSources) {
      baselineCoverageInstrumentedFilesBuilder.addTransitive(localSources);
    }

    void collectLocalMetadata(
        LocalMetadataCollector localMetadataCollector, Iterable<Artifact> rootFiles) {
      localMetadataCollector.collectMetadataArtifacts(
          rootFiles, ruleContext.getAnalysisEnvironment(), metadataFilesBuilder);
    }

    InstrumentedFilesInfo build() {
      NestedSet<Artifact> baselineCoverageFiles = baselineCoverageInstrumentedFilesBuilder.build();
      return new InstrumentedFilesInfo(
          instrumentedFilesBuilder.build(),
          metadataFilesBuilder.build(),
          baselineCoverageFiles,
          // Create one baseline coverage action per target, for the transitive closure of files.
          BaselineCoverageAction.create(ruleContext, baselineCoverageFiles),
          coverageSupportFilesBuilder.build(),
          coverageEnvironmentBuilder.build(),
          reportedToActualSources);
    }
  }

  /**
   * An explicit constant for a {@link LocalMetadataCollector} that doesn't collect anything.
   */
  public static final LocalMetadataCollector NO_METADATA_COLLECTOR = null;

  private static Iterable<TransitiveInfoCollection> getPrerequisitesForAttributes(
      RuleContext ruleContext, Collection<String> attributeNames) {
    List<TransitiveInfoCollection> prerequisites = new ArrayList<>();
    for (String attr : attributeNames) {
      if (ruleContext.getRule().isAttrDefined(attr, BuildType.LABEL_LIST) ||
          ruleContext.getRule().isAttrDefined(attr, BuildType.LABEL)) {
        prerequisites.addAll(ruleContext.getPrerequisites(attr, TransitionMode.DONT_CHECK));
      }
    }
    return prerequisites;
  }

  private static Iterable<TransitiveInfoCollection> getAllNonToolPrerequisites(
      RuleContext ruleContext) {
    List<TransitiveInfoCollection> prerequisites = new ArrayList<>();
    for (Attribute attr : ruleContext.getRule().getAttributes()) {
      if ((attr.getType() == BuildType.LABEL_LIST || attr.getType() == BuildType.LABEL)
          && !attr.getTransitionFactory().isTool()) {
        prerequisites.addAll(
            ruleContext.getPrerequisites(attr.getName(), TransitionMode.DONT_CHECK));
      }
    }
    return prerequisites;
  }
}
