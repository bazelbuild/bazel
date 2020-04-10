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

package com.google.devtools.build.lib.analysis;

import static com.google.devtools.build.lib.buildeventstream.TestFileNameConstants.BASELINE_COVERAGE;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.ArtifactReceiver;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.File;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.OutputGroup;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TargetComplete;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.function.Function;
import javax.annotation.Nullable;

/** This event is fired as soon as a target is either built or fails. */
public final class TargetCompleteEvent
    implements SkyValue,
        BuildEventWithOrderConstraint,
        EventReportingArtifacts,
        BuildEventWithConfiguration {

  /** Lightweight data needed about the configured target in this event. */
  public static class ExecutableTargetData {
    @Nullable private final Path runfilesDirectory;
    @Nullable private final Artifact executable;

    private ExecutableTargetData(ConfiguredTargetAndData targetAndData) {
      FilesToRunProvider provider =
          targetAndData.getConfiguredTarget().getProvider(FilesToRunProvider.class);
      if (provider != null) {
        this.executable = provider.getExecutable();
        if (null != provider.getRunfilesSupport()) {
          this.runfilesDirectory = provider.getRunfilesSupport().getRunfilesDirectory();
        } else {
          this.runfilesDirectory = null;
        }
      } else {
        this.executable = null;
        this.runfilesDirectory = null;
      }
    }

    @Nullable
    public Path getRunfilesDirectory() {
      return runfilesDirectory;
    }

    @Nullable
    public Artifact getExecutable() {
      return executable;
    }
  }

  private final Label label;
  private final ConfiguredTargetKey configuredTargetKey;
  private final NestedSet<Cause> rootCauses;
  private final ImmutableList<BuildEventId> postedAfter;
  private final CompletionContext completionContext;
  private final NestedSet<ArtifactsInOutputGroup> outputs;
  private final NestedSet<Artifact> baselineCoverageArtifacts;
  private final Label aliasLabel;
  private final boolean isTest;
  @Nullable private final Long testTimeoutSeconds;
  @Nullable private final TestProvider.TestParams testParams;
  private final BuildEvent configurationEvent;
  private final BuildEventId configEventId;
  private final Iterable<String> tags;
  private final ExecutableTargetData executableTargetData;
  @Nullable private final DetailedExitCode detailedExitCode;

  private TargetCompleteEvent(
      ConfiguredTargetAndData targetAndData,
      NestedSet<Cause> rootCauses,
      CompletionContext completionContext,
      NestedSet<ArtifactsInOutputGroup> outputs,
      boolean isTest) {
    this.rootCauses =
        (rootCauses == null) ? NestedSetBuilder.emptySet(Order.STABLE_ORDER) : rootCauses;
    this.executableTargetData = new ExecutableTargetData(targetAndData);
    ImmutableList.Builder<BuildEventId> postedAfterBuilder = ImmutableList.builder();
    this.label = targetAndData.getConfiguredTarget().getLabel();
    this.aliasLabel = targetAndData.getConfiguredTarget().getOriginalLabel();
    this.configuredTargetKey =
        ConfiguredTargetKey.of(
            targetAndData.getConfiguredTarget(), targetAndData.getConfiguration());
    postedAfterBuilder.add(BuildEventIdUtil.targetConfigured(aliasLabel));
    DetailedExitCode mostImportantDetailedExitCode = null;
    for (Cause cause : getRootCauses().toList()) {
      mostImportantDetailedExitCode =
          DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
              mostImportantDetailedExitCode, cause.getDetailedExitCode());
      postedAfterBuilder.add(cause.getIdProto());
    }
    detailedExitCode = mostImportantDetailedExitCode;
    this.postedAfter = postedAfterBuilder.build();
    this.completionContext = completionContext;
    this.outputs = outputs;
    this.isTest = isTest;
    this.testTimeoutSeconds = isTest ? getTestTimeoutSeconds(targetAndData) : null;
    BuildConfiguration configuration = targetAndData.getConfiguration();
    this.configEventId =
        configuration != null ? configuration.getEventId() : BuildEventIdUtil.nullConfigurationId();
    this.configurationEvent = configuration != null ? configuration.toBuildEvent() : null;
    this.testParams =
        isTest
            ? targetAndData.getConfiguredTarget().getProvider(TestProvider.class).getTestParams()
            : null;
    InstrumentedFilesInfo instrumentedFilesProvider =
        targetAndData.getConfiguredTarget().get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR);
    if (instrumentedFilesProvider == null) {
      this.baselineCoverageArtifacts = null;
    } else {
      NestedSet<Artifact> baselineCoverageArtifacts =
          instrumentedFilesProvider.getBaselineCoverageArtifacts();
      if (!baselineCoverageArtifacts.isEmpty()) {
        this.baselineCoverageArtifacts = baselineCoverageArtifacts;
      } else {
        this.baselineCoverageArtifacts = null;
      }
    }
    // For tags, we are only interested in targets that are rules.
    if (!(targetAndData.getConfiguredTarget() instanceof RuleConfiguredTarget)) {
      this.tags = ImmutableList.of();
    } else {
      AttributeMap attributes =
          ConfiguredAttributeMapper.of(
              (Rule) targetAndData.getTarget(),
              ((RuleConfiguredTarget) targetAndData.getConfiguredTarget()).getConfigConditions());
      // Every build rule (implicitly) has a "tags" attribute. However other rule configured targets
      // are repository rules (which don't have a tags attribute); morevoer, thanks to the virtual
      // "external" package, they are user visible as targets and can create a completed event as
      // well.
      if (attributes.has("tags", Type.STRING_LIST)) {
        this.tags = attributes.get("tags", Type.STRING_LIST);
      } else {
        this.tags = ImmutableList.of();
      }
    }
  }

  /** Construct a successful target completion event. */
  public static TargetCompleteEvent successfulBuild(
      ConfiguredTargetAndData ct,
      CompletionContext completionContext,
      NestedSet<ArtifactsInOutputGroup> outputs) {
    return new TargetCompleteEvent(ct, null, completionContext, outputs, false);
  }

  /** Construct a successful target completion event for a target that will be tested. */
  public static TargetCompleteEvent successfulBuildSchedulingTest(
      ConfiguredTargetAndData ct,
      CompletionContext completionContext,
      NestedSet<ArtifactsInOutputGroup> outputs) {
    return new TargetCompleteEvent(ct, null, completionContext, outputs, true);
  }

  /**
   * Construct a target completion event for a failed target, with the given non-empty root causes.
   */
  public static TargetCompleteEvent createFailed(
      ConfiguredTargetAndData ct,
      NestedSet<Cause> rootCauses,
      NestedSet<ArtifactsInOutputGroup> outputs) {
    Preconditions.checkArgument(!rootCauses.isEmpty());
    return new TargetCompleteEvent(
        ct, rootCauses, CompletionContext.FAILED_COMPLETION_CTX, outputs, false);
  }

  /** Returns the label of the target associated with the event. */
  public Label getLabel() {
    return label;
  }

  public ConfiguredTargetKey getConfiguredTargetKey() {
    return configuredTargetKey;
  }

  public ExecutableTargetData getExecutableTargetData() {
    return executableTargetData;
  }

  /** Determines whether the target has failed or succeeded. */
  public boolean failed() {
    return !rootCauses.isEmpty();
  }

  /** Get the root causes of the target. May be empty. */
  public NestedSet<Cause> getRootCauses() {
    return rootCauses;
  }

  public Iterable<Artifact> getLegacyFilteredImportantArtifacts() {
    // TODO(ulfjack): This duplicates code in ArtifactsToBuild.
    NestedSetBuilder<Artifact> builder = new NestedSetBuilder<>(outputs.getOrder());
    for (ArtifactsInOutputGroup artifactsInOutputGroup : outputs.toList()) {
      if (artifactsInOutputGroup.areImportant()) {
        builder.addTransitive(artifactsInOutputGroup.getArtifacts());
      }
    }
    return Iterables.filter(
        builder.build().toList(),
        (artifact) -> !artifact.isSourceArtifact() && !artifact.isMiddlemanArtifact());
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.targetCompleted(aliasLabel, configEventId);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder<BuildEventId> childrenBuilder = ImmutableList.builder();
    for (Cause cause : getRootCauses().toList()) {
      childrenBuilder.add(cause.getIdProto());
    }
    if (isTest) {
      // For tests, announce all the test actions that will minimally happen (except for
      // interruption). If after the result of a test action another attempt is necessary,
      // it will be announced with the action that made the new attempt necessary.
      Label label = getLabel();
      for (int run = 0; run < Math.max(testParams.getRuns(), 1); run++) {
        for (int shard = 0; shard < Math.max(testParams.getShards(), 1); shard++) {
          childrenBuilder.add(BuildEventIdUtil.testResult(label, run, shard, configEventId));
        }
      }
      childrenBuilder.add(BuildEventIdUtil.testSummary(label, configEventId));
    }
    return childrenBuilder.build();
  }

  // TODO(aehlig): remove as soon as we managed to get rid of the deprecated "important_output"
  // field.
  private static void addImportantOutputs(
      CompletionContext completionContext,
      TargetComplete.Builder builder,
      BuildEventContext converters,
      Iterable<Artifact> artifacts) {
    addImportantOutputs(
        completionContext, builder, Artifact::getRootRelativePathString, converters, artifacts);
  }

  private static Iterable<Artifact> filterFilesets(Iterable<Artifact> artifacts) {
    return Iterables.filter(artifacts, artifact -> !artifact.isFileset());
  }

  private static void addImportantOutputs(
      CompletionContext completionContext,
      BuildEventStreamProtos.TargetComplete.Builder builder,
      Function<Artifact, String> artifactNameFunction,
      BuildEventContext converters,
      Iterable<Artifact> artifacts) {
    completionContext.visitArtifacts(
        filterFilesets(artifacts),
        new ArtifactReceiver() {
          @Override
          public void accept(Artifact artifact) {
            String name = artifactNameFunction.apply(artifact);
            String uri =
                converters.pathConverter().apply(completionContext.pathResolver().toPath(artifact));
            if (uri != null) {
              builder.addImportantOutput(newFileFromArtifact(name, artifact).setUri(uri).build());
            }
          }

          @Override
          public void acceptFilesetMapping(
              Artifact fileset, PathFragment relativePath, Path targetFile) {
            throw new IllegalStateException(fileset + " should have been filtered out");
          }
        });
  }

  public static BuildEventStreamProtos.File.Builder newFileFromArtifact(
      String name, Artifact artifact) {
    return newFileFromArtifact(name, artifact, PathFragment.EMPTY_FRAGMENT);
  }

  public static BuildEventStreamProtos.File.Builder newFileFromArtifact(
      String name, Artifact artifact, PathFragment relPath) {
    File.Builder builder =
        File.newBuilder()
            .setName(
                name == null
                    ? artifact.getRootRelativePath().getRelative(relPath).getPathString()
                    : name);
    builder.addAllPathPrefix(artifact.getRoot().getComponents());
    return builder;
  }

  public static BuildEventStreamProtos.File.Builder newFileFromArtifact(Artifact artifact) {
    return newFileFromArtifact(/* name= */ null, artifact);
  }

  @Override
  public Collection<LocalFile> referencedLocalFiles() {
    ImmutableList.Builder<LocalFile> builder = ImmutableList.builder();
    for (ArtifactsInOutputGroup group : outputs.toList()) {
      if (group.areImportant()) {
        completionContext.visitArtifacts(
            filterFilesets(group.getArtifacts().toList()),
            new ArtifactReceiver() {
              @Override
              public void accept(Artifact artifact) {
                builder.add(
                    new LocalFile(
                        completionContext.pathResolver().toPath(artifact), LocalFileType.OUTPUT));
              }

              @Override
              public void acceptFilesetMapping(
                  Artifact fileset, PathFragment name, Path targetFile) {
                throw new IllegalStateException(fileset + " should have been filtered out");
              }
            });
      }
    }
    if (baselineCoverageArtifacts != null) {
      for (Artifact artifact : baselineCoverageArtifacts.toList()) {
        builder.add(
            new LocalFile(
                completionContext.pathResolver().toPath(artifact), LocalFileType.COVERAGE_OUTPUT));
      }
    }
    return builder.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.TargetComplete.Builder builder =
        BuildEventStreamProtos.TargetComplete.newBuilder();

    boolean failed = failed();
    builder.setSuccess(!failed);
    if (detailedExitCode != null) {
      if (!failed) {
        BugReport.sendBugReport(
            new IllegalStateException("Detailed exit code with success? " + detailedExitCode));
      }
      FailureDetails.FailureDetail failureDetail = detailedExitCode.getFailureDetail();
      if (failureDetail != null) {
        builder.setFailureDetail(failureDetail);
      }
    }
    builder.addAllTag(getTags());
    builder.addAllOutputGroup(getOutputFilesByGroup(converters.artifactGroupNamer()));

    if (isTest) {
      builder.setTestTimeoutSeconds(testTimeoutSeconds);
    }

    Iterable<Artifact> filteredImportantArtifacts = getLegacyFilteredImportantArtifacts();
    for (Artifact artifact : filteredImportantArtifacts) {
      if (artifact.isDirectory()) {
        builder.addDirectoryOutput(newFileFromArtifact(artifact).build());
      }
    }
    // TODO(aehlig): remove direct reporting of artifacts as soon as clients no longer
    // need it.
    if (converters.getOptions().legacyImportantOutputs) {
      addImportantOutputs(completionContext, builder, converters, filteredImportantArtifacts);
      if (baselineCoverageArtifacts != null) {
        addImportantOutputs(
            completionContext,
            builder,
            (artifact -> BASELINE_COVERAGE),
            converters,
            baselineCoverageArtifacts.toList());
      }
    }

    BuildEventStreamProtos.TargetComplete complete = builder.build();
    return GenericBuildEvent.protoChaining(this).setCompleted(complete).build();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return postedAfter;
  }

  @Override
  public ReportedArtifacts reportedArtifacts() {
    ImmutableSet.Builder<NestedSet<Artifact>> builder = ImmutableSet.builder();
    for (ArtifactsInOutputGroup artifactsInGroup : outputs.toList()) {
      if (artifactsInGroup.areImportant()) {
        builder.add(artifactsInGroup.getArtifacts());
      }
    }
    if (baselineCoverageArtifacts != null) {
      builder.add(baselineCoverageArtifacts);
    }
    return new ReportedArtifacts(builder.build(), completionContext);
  }

  @Override
  public Collection<BuildEvent> getConfigurations() {
    return configurationEvent != null ? ImmutableList.of(configurationEvent) : ImmutableList.of();
  }

  private Iterable<String> getTags() {
    return tags;
  }

  private Iterable<OutputGroup> getOutputFilesByGroup(ArtifactGroupNamer namer) {
    ImmutableList.Builder<OutputGroup> groups = ImmutableList.builder();
    for (ArtifactsInOutputGroup artifactsInOutputGroup : outputs.toList()) {
      if (!artifactsInOutputGroup.areImportant()) {
        continue;
      }
      OutputGroup.Builder groupBuilder = OutputGroup.newBuilder();
      groupBuilder.setName(artifactsInOutputGroup.getOutputGroup());
      groupBuilder.addFileSets(
          namer.apply(
              (new NestedSetView<Artifact>(artifactsInOutputGroup.getArtifacts())).identifier()));
      groups.add(groupBuilder.build());
    }
    if (baselineCoverageArtifacts != null) {
      groups.add(
          OutputGroup.newBuilder()
              .setName(BASELINE_COVERAGE)
              .addFileSets(
                  namer.apply(
                      (new NestedSetView<Artifact>(baselineCoverageArtifacts).identifier())))
              .build());
    }
    return groups.build();
  }

  /**
   * Returns timeout value in seconds that should be used for all test actions under this configured
   * target. We always use the "categorical timeouts" which are based on the --test_timeout flag. A
   * rule picks its timeout but ends up with the same effective value as all other rules in that
   * category and configuration.
   */
  private static Long getTestTimeoutSeconds(ConfiguredTargetAndData targetAndData) {
    BuildConfiguration configuration = targetAndData.getConfiguration();
    Rule associatedRule = targetAndData.getTarget().getAssociatedRule();
    TestTimeout categoricalTimeout = TestTimeout.getTestTimeout(associatedRule);
    return configuration
        .getFragment(TestConfiguration.class)
        .getTestTimeout()
        .get(categoricalTimeout)
        .getSeconds();
  }
}
