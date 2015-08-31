// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules.test;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import javax.annotation.Nullable;

/**
 * A helper class for collecting instrumented files and metadata for a target.
 */
public final class InstrumentedFilesCollector {
  public static InstrumentedFilesProvider collect(RuleContext ruleContext, InstrumentationSpec spec,
      @Nullable LocalMetadataCollector localMetadataCollector,
      @Nullable Iterable<Artifact> rootFiles) {
    return collect(ruleContext, spec, localMetadataCollector, rootFiles, false);
  }

  /**
   * Collects transitive instrumentation data from dependencies, collects local source files from
   * dependencies, collects local metadata files by traversing the action graph of the current
   * configured target, and creates baseline coverage actions for the transitive closure of source
   * files (if <code>withBaselineCoverage</code> is true).
   */
  public static InstrumentedFilesProvider collect(RuleContext ruleContext,
      InstrumentationSpec spec, @Nullable LocalMetadataCollector localMetadataCollector,
      @Nullable Iterable<Artifact> rootFiles, boolean withBaselineCoverage) {
    Preconditions.checkNotNull(ruleContext);
    Preconditions.checkNotNull(spec);

    if (!ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      return InstrumentedFilesProviderImpl.EMPTY;
    }

    NestedSetBuilder<Artifact> instrumentedFilesBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> metadataFilesBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> baselineCoverageArtifactsBuilder = NestedSetBuilder.stableOrder();

    Iterable<TransitiveInfoCollection> prereqs = getAllPrerequisites(ruleContext, spec);

    // Transitive instrumentation data.
    for (TransitiveInfoCollection dep : prereqs) {
      InstrumentedFilesProvider provider = dep.getProvider(InstrumentedFilesProvider.class);
      if (provider != null) {
        instrumentedFilesBuilder.addTransitive(provider.getInstrumentedFiles());
        metadataFilesBuilder.addTransitive(provider.getInstrumentationMetadataFiles());
        baselineCoverageArtifactsBuilder.addTransitive(provider.getBaselineCoverageArtifacts());
      }
    }

    // Local sources.
    NestedSet<Artifact> localSources = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    if (shouldIncludeLocalSources(ruleContext)) {
      NestedSetBuilder<Artifact> localSourcesBuilder = NestedSetBuilder.stableOrder();
      for (TransitiveInfoCollection dep : prereqs) {
        for (Artifact artifact : dep.getProvider(FileProvider.class).getFilesToBuild()) {
          if (artifact.isSourceArtifact() &&
              spec.instrumentedFileTypes.matches(artifact.getFilename())) {
            localSourcesBuilder.add(artifact);
          }
        }
      }
      localSources = localSourcesBuilder.build();
    }
    instrumentedFilesBuilder.addTransitive(localSources);

    // Local metadata files.
    if (localMetadataCollector != null) {
      localMetadataCollector.collectMetadataArtifacts(rootFiles,
          ruleContext.getAnalysisEnvironment(), metadataFilesBuilder);
    }

    // Baseline coverage actions.
    if (withBaselineCoverage) {
      baselineCoverageArtifactsBuilder.addTransitive(
          BaselineCoverageAction.getBaselineCoverageArtifacts(ruleContext, localSources));
    }
    return new InstrumentedFilesProviderImpl(instrumentedFilesBuilder.build(),
        metadataFilesBuilder.build(),
        baselineCoverageArtifactsBuilder.build(),
        ImmutableMap.<String, String>of());
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

    /**
     * The list of attributes which should be (transitively) checked for sources and instrumentation
     * metadata.
     */
    private final Collection<String> instrumentedAttributes;

    public InstrumentationSpec(FileTypeSet instrumentedFileTypes,
        Collection<String> instrumentedAttributes) {
      this.instrumentedFileTypes = instrumentedFileTypes;
      this.instrumentedAttributes = ImmutableList.copyOf(instrumentedAttributes);
    }

    public InstrumentationSpec(FileTypeSet instrumentedFileTypes,
        String... instrumentedAttributes) {
      this(instrumentedFileTypes, ImmutableList.copyOf(instrumentedAttributes));
    }

    /**
     * Returns a new instrumentation spec with the given attribute names replacing the ones
     * stored in this object.
     */
    public InstrumentationSpec withAttributes(String... instrumentedAttributes) {
      return new InstrumentationSpec(instrumentedFileTypes, instrumentedAttributes);
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
                              Action action, FileType fileType) {
      for (Artifact output : action.getOutputs()) {
        if (fileType.matches(output.getFilename())) {
          metadataFilesBuilder.add(output);
          break;
        }
      }
    }
  }

  /**
   * Only collects files transitively from srcs, deps, and data attributes.
   */
  public static final InstrumentationSpec TRANSITIVE_COLLECTION_SPEC = new InstrumentationSpec(
      FileTypeSet.NO_FILE,
      "srcs", "deps", "data");

  /**
   * An explicit constant for a {@link LocalMetadataCollector} that doesn't collect anything.
   */
  public static final LocalMetadataCollector NO_METADATA_COLLECTOR = null;

  private static boolean shouldIncludeLocalSources(RuleContext ruleContext) {
    return ruleContext.getConfiguration().getInstrumentationFilter().isIncluded(
        ruleContext.getLabel().toString());
  }

  private static Iterable<TransitiveInfoCollection> getAllPrerequisites(
      RuleContext ruleContext, InstrumentationSpec spec) {
    List<TransitiveInfoCollection> prerequisites = new ArrayList<>();
    for (String attr : spec.instrumentedAttributes) {
      if (ruleContext.getRule().isAttrDefined(attr, Type.LABEL_LIST) ||
          ruleContext.getRule().isAttrDefined(attr, Type.LABEL)) {
        prerequisites.addAll(ruleContext.getPrerequisites(attr, Mode.DONT_CHECK));
      }
    }
    return prerequisites;
  }
}
