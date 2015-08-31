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

/**
 * A helper class for collecting instrumented files and metadata for a target.
 */
public final class InstrumentedFilesCollector {
  public static InstrumentedFilesProvider collect(RuleContext ruleContext, InstrumentationSpec spec,
      LocalMetadataCollector localMetadataCollector, Iterable<Artifact> rootFiles) {
    return collect(ruleContext, spec, localMetadataCollector, rootFiles, false);
  }

  public static InstrumentedFilesProvider collect(RuleContext ruleContext,
      InstrumentationSpec spec, LocalMetadataCollector localMetadataCollector,
      Iterable<Artifact> rootFiles, boolean withBaselineCoverage) {
    InstrumentedFilesCollector collector = new InstrumentedFilesCollector(ruleContext, spec,
        localMetadataCollector, rootFiles);
    NestedSet<Artifact> baselineCoverageArtifacts;
    if (withBaselineCoverage) {
      baselineCoverageArtifacts =
          BaselineCoverageAction.getBaselineCoverageArtifacts(ruleContext,
              collector.instrumentedFiles);
    } else {
      baselineCoverageArtifacts = NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);
    }
    return new InstrumentedFilesProviderImpl(collector.instrumentedFiles,
        collector.instrumentationMetadataFiles,
        baselineCoverageArtifacts,
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

  private final RuleContext ruleContext;
  private final InstrumentationSpec spec;
  private final LocalMetadataCollector localMetadataCollector;
  private final NestedSet<Artifact> instrumentationMetadataFiles;
  private final NestedSet<Artifact> instrumentedFiles;

  private InstrumentedFilesCollector(RuleContext ruleContext, InstrumentationSpec spec,
      LocalMetadataCollector localMetadataCollector, Iterable<Artifact> rootFiles) {
    this.ruleContext = ruleContext;
    this.spec = spec;
    this.localMetadataCollector = localMetadataCollector;
    Preconditions.checkNotNull(ruleContext, "RuleContext already cleared. That means that the"
        + " collector data was already memoized. You do not have to call it again.");
    if (!ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      instrumentedFiles = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      instrumentationMetadataFiles = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    } else {
      NestedSetBuilder<Artifact> instrumentedFilesBuilder =
          NestedSetBuilder.stableOrder();
      NestedSetBuilder<Artifact> metadataFilesBuilder = NestedSetBuilder.stableOrder();
      collect(ruleContext.getAnalysisEnvironment(), instrumentedFilesBuilder, metadataFilesBuilder,
          rootFiles);
      instrumentedFiles = instrumentedFilesBuilder.build();
      instrumentationMetadataFiles = metadataFilesBuilder.build();
    }
  }

  /**
   * Returns instrumented source files for the target provided during construction.
   */
  public final NestedSet<Artifact> getInstrumentedFiles() {
    return instrumentedFiles;
  }

  /**
   * Returns instrumentation metadata files for the target provided during construction.
   */
  public final NestedSet<Artifact> getInstrumentationMetadataFiles() {
    return instrumentationMetadataFiles;
  }

  /**
   * Collects instrumented files and metadata files.
   */
  private void collect(AnalysisEnvironment analysisEnvironment,
      NestedSetBuilder<Artifact> instrumentedFilesBuilder,
      NestedSetBuilder<Artifact> metadataFilesBuilder,
      Iterable<Artifact> rootFiles) {
    for (TransitiveInfoCollection dep : getAllPrerequisites()) {
      InstrumentedFilesProvider provider = dep.getProvider(InstrumentedFilesProvider.class);
      if (provider != null) {
        instrumentedFilesBuilder.addTransitive(provider.getInstrumentedFiles());
        metadataFilesBuilder.addTransitive(provider.getInstrumentationMetadataFiles());
      } else if (shouldIncludeLocalSources()) {
        for (Artifact artifact : dep.getProvider(FileProvider.class).getFilesToBuild()) {
          if (artifact.isSourceArtifact() &&
              spec.instrumentedFileTypes.matches(artifact.getFilename())) {
            instrumentedFilesBuilder.add(artifact);
          }
        }
      }
    }

    if (localMetadataCollector != null) {
      localMetadataCollector.collectMetadataArtifacts(rootFiles,
          analysisEnvironment, metadataFilesBuilder);
    }
  }

  /**
   * Returns the list of attributes which should be (transitively) checked for sources and
   * instrumentation metadata.
   */
  private Collection<String> getSourceAttributes() {
    return spec.instrumentedAttributes;
  }

  private boolean shouldIncludeLocalSources() {
    return ruleContext.getConfiguration().getInstrumentationFilter().isIncluded(
        ruleContext.getLabel().toString());
  }

  private Iterable<TransitiveInfoCollection> getAllPrerequisites() {
    List<TransitiveInfoCollection> prerequisites = new ArrayList<>();
    for (String attr : getSourceAttributes()) {
      if (ruleContext.getRule().isAttrDefined(attr, Type.LABEL_LIST) ||
          ruleContext.getRule().isAttrDefined(attr, Type.LABEL)) {
        prerequisites.addAll(ruleContext.getPrerequisites(attr, Mode.DONT_CHECK));
      }
    }
    return prerequisites;
  }
}
