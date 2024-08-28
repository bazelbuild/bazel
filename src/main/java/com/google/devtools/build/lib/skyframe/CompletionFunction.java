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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.InputFileErrorException;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.SuccessfulArtifactFilter;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingArtifactValue;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.SourceArtifactException;
import com.google.devtools.build.lib.skyframe.MetadataConsumerForMetrics.FilesMetricConsumer;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** CompletionFunction builds the artifactsToBuild collection of a {@link ConfiguredTarget}. */
public final class CompletionFunction<
        ValueT extends ConfiguredObjectValue,
        ResultT extends SkyValue,
        KeyT extends TopLevelActionLookupKeyWrapper,
        FailureT>
    implements SkyFunction {

  /** A strategy for completing the build. */
  interface Completor<
      ValueT, ResultT extends SkyValue, KeyT extends TopLevelActionLookupKeyWrapper, FailureT> {

    /** Creates an event reporting an absent input artifact. */
    Event getRootCauseError(ValueT value, KeyT key, LabelCause rootCause, Environment env)
        throws InterruptedException;

    @Nullable
    Object getLocationIdentifier(ValueT value, KeyT key, Environment env)
        throws InterruptedException;

    /** Provides a successful completion value. */
    ResultT getResult();

    /**
     * Creates supplementary data needed to call {@link #createFailed(Object, NestedSet,
     * CompletionContext, ImmutableMap, Object)}; returns null if skyframe found missing values.
     */
    @Nullable
    FailureT getFailureData(KeyT key, ValueT value, Environment env) throws InterruptedException;

    /** Creates a failed completion value. */
    ExtendedEventHandler.Postable createFailed(
        KeyT skyKey,
        NestedSet<Cause> rootCauses,
        CompletionContext ctx,
        ImmutableMap<String, ArtifactsInOutputGroup> outputs,
        FailureT failureData)
        throws InterruptedException;

    /** Creates a succeeded completion value; returns null if skyframe found missing values. */
    @Nullable
    EventReportingArtifacts createSucceeded(
        KeyT skyKey,
        ValueT value,
        CompletionContext completionContext,
        ArtifactsToBuild artifactsToBuild,
        Environment env)
        throws InterruptedException;
  }

  private final PathResolverFactory pathResolverFactory;
  private final Completor<ValueT, ResultT, KeyT, FailureT> completor;
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final FilesMetricConsumer topLevelArtifactsMetric;
  private final BugReporter bugReporter;
  private final Supplier<Boolean> isSkymeld;

  CompletionFunction(
      PathResolverFactory pathResolverFactory,
      Completor<ValueT, ResultT, KeyT, FailureT> completor,
      SkyframeActionExecutor skyframeActionExecutor,
      FilesMetricConsumer topLevelArtifactsMetric,
      BugReporter bugReporter,
      Supplier<Boolean> isSkymeld) {
    this.pathResolverFactory = pathResolverFactory;
    this.completor = completor;
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.topLevelArtifactsMetric = topLevelArtifactsMetric;
    this.bugReporter = bugReporter;
    this.isSkymeld = isSkymeld;
  }

  @SuppressWarnings("unchecked") // Cast to KeyT
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws CompletionFunctionException, InterruptedException {
    WorkspaceNameValue workspaceNameValue =
        (WorkspaceNameValue) env.getValue(WorkspaceNameValue.key());
    if (workspaceNameValue == null) {
      return null;
    }

    KeyT key = (KeyT) skyKey;
    Pair<ValueT, ArtifactsToBuild> valueAndArtifactsToBuild = getValueAndArtifactsToBuild(key, env);
    if (env.valuesMissing()) {
      return null;
    }
    ValueT value = valueAndArtifactsToBuild.first;
    ArtifactsToBuild artifactsToBuild = valueAndArtifactsToBuild.second;

    // Ensure that coverage artifacts are built before a target is considered completed.
    ImmutableList<Artifact> allArtifacts = artifactsToBuild.getAllArtifacts().toList();
    InstrumentedFilesInfo instrumentedFilesInfo =
        value.getConfiguredObject().get(InstrumentedFilesInfo.STARLARK_CONSTRUCTOR);
    Iterable<SkyKey> keysToRequest;
    if (value.getConfiguredObject() instanceof ConfiguredTarget && instrumentedFilesInfo != null) {
      keysToRequest =
          Iterables.concat(
              Artifact.keys(allArtifacts),
              Artifact.keys(instrumentedFilesInfo.getBaselineCoverageArtifacts().toList()));
    } else {
      keysToRequest = Artifact.keys(allArtifacts);
    }
    SkyframeLookupResult inputDeps = env.getValuesAndExceptions(keysToRequest);

    boolean allArtifactsAreImportant = artifactsToBuild.areAllOutputGroupsImportant();

    ActionInputMap inputMap = new ActionInputMap(bugReporter, allArtifacts.size());
    // Prepare an ActionInputMap for important artifacts separately, to be used by BEP events. The
    // _validation output group can contain orders of magnitude more unimportant artifacts than
    // there are important artifacts, and BEP events will retain the ActionInputMap until the
    // event is delivered to transports. If the BEP events reference *all* artifacts it can increase
    // heap high-watermark by multiple GB.
    ActionInputMap importantInputMap;
    Set<Artifact> importantArtifactSet;
    if (allArtifactsAreImportant) {
      importantArtifactSet = ImmutableSet.of();
      importantInputMap = inputMap;
    } else {
      ImmutableList<Artifact> importantArtifacts =
          artifactsToBuild.getImportantArtifacts().toList();
      importantArtifactSet = new HashSet<>(importantArtifacts);
      importantInputMap = new ActionInputMap(bugReporter, importantArtifacts.size());
    }

    Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets = new HashMap<>();
    Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets = new HashMap<>();

    ActionExecutionException firstActionExecutionException = null;
    NestedSetBuilder<Cause> rootCausesBuilder = NestedSetBuilder.stableOrder();
    ImmutableSet.Builder<Artifact> builtArtifactsBuilder = ImmutableSet.builder();
    // Don't double-count files due to Skyframe restarts.
    FilesMetricConsumer currentConsumer = new FilesMetricConsumer();
    for (Artifact input : allArtifacts) {
      try {
        SkyValue artifactValue =
            inputDeps.getOrThrow(
                Artifact.key(input), ActionExecutionException.class, SourceArtifactException.class);
        if (artifactValue != null) {
          if (artifactValue instanceof MissingArtifactValue) {
            handleSourceFileError(
                input,
                ((MissingArtifactValue) artifactValue).getDetailedExitCode(),
                rootCausesBuilder,
                env,
                value,
                key);
          } else {
            builtArtifactsBuilder.add(input);
            ActionInputMapHelper.addToMap(
                inputMap,
                expandedArtifacts,
                archivedTreeArtifacts,
                expandedFilesets,
                topLevelFilesets,
                input,
                artifactValue,
                env,
                currentConsumer);
            if (!allArtifactsAreImportant && importantArtifactSet.contains(input)) {
              // Calling #addToMap a second time with `input` and `artifactValue` will perform no-op
              // updates to the secondary collections passed in (eg. expandedArtifacts,
              // topLevelFilesets). MetadataConsumerForMetrics.NO_OP is used to avoid
              // double-counting.
              ActionInputMapHelper.addToMap(
                  importantInputMap,
                  expandedArtifacts,
                  archivedTreeArtifacts,
                  expandedFilesets,
                  topLevelFilesets,
                  input,
                  artifactValue,
                  env);
            }
          }
        }
      } catch (ActionExecutionException e) {
        rootCausesBuilder.addTransitive(e.getRootCauses());
        // Prefer a catastrophic exception as the one we propagate.
        if (firstActionExecutionException == null
            || (!firstActionExecutionException.isCatastrophe() && e.isCatastrophe())) {
          firstActionExecutionException = e;
        }
      } catch (SourceArtifactException e) {
        if (!input.isSourceArtifact()) {
          bugReporter.logUnexpected(
              e, "Non-source artifact had SourceArtifactException: %s", input);
        }
        handleSourceFileError(input, e.getDetailedExitCode(), rootCausesBuilder, env, value, key);
      }
    }
    expandedFilesets.putAll(topLevelFilesets);

    NestedSet<Cause> rootCauses = rootCausesBuilder.build();
    @Nullable FailureT failureData = null;
    if (!rootCauses.isEmpty()) {
      failureData = completor.getFailureData(key, value, env);
      if (failureData == null) {
        return null;
      }
    }

    CompletionContext ctx =
        CompletionContext.create(
            expandedArtifacts,
            expandedFilesets,
            key.topLevelArtifactContext().expandFilesets(),
            key.topLevelArtifactContext().fullyResolveFilesetSymlinks(),
            inputMap,
            importantInputMap,
            pathResolverFactory,
            skyframeActionExecutor.getExecRoot(),
            workspaceNameValue.getName());

    if (!rootCauses.isEmpty()) {
      ImmutableMap<String, ArtifactsInOutputGroup> builtOutputs =
          new SuccessfulArtifactFilter(builtArtifactsBuilder.build())
              .filterArtifactsInOutputGroup(artifactsToBuild.getAllArtifactsByOutputGroup());
      env.getListener()
          .post(completor.createFailed(key, rootCauses, ctx, builtOutputs, failureData));
      if (firstActionExecutionException != null) {
        throw new CompletionFunctionException(firstActionExecutionException);
      }
      // locationPrefix theoretically *could* be null because of missing deps, but not in reality,
      // and we're not allowed to wait for deps to be ready if we're failing anyway.
      @Nullable Object locationPrefix = completor.getLocationIdentifier(value, key, env);
      Pair<DetailedExitCode, String> codeAndMessage =
          ActionExecutionFunction.createSourceErrorCodeAndMessage(rootCauses.toList(), key);
      String message;
      if (locationPrefix instanceof Location) {
        message = codeAndMessage.getSecond();
        env.getListener().handle(Event.error((Location) locationPrefix, message));
      } else {
        message = (locationPrefix == null ? "" : locationPrefix + " ") + codeAndMessage.getSecond();
        env.getListener().handle(Event.error(message));
      }
      throw new CompletionFunctionException(
          new InputFileErrorException(message, codeAndMessage.getFirst()));
    }

    // Only check for missing values *after* reporting errors: if there are missing files in a build
    // with --nokeep_going, there may be missing dependencies during error bubbling, we still need
    // to report the error.
    if (env.valuesMissing()) {
      return null;
    }

    ExtendedEventHandler.Postable postable =
        completor.createSucceeded(key, value, ctx, artifactsToBuild, env);
    if (postable == null) {
      return null;
    }
    ensureToplevelArtifacts(env, postable, inputMap);
    env.getListener().post(postable);
    topLevelArtifactsMetric.mergeIn(currentConsumer);

    return completor.getResult();
  }

  private void ensureToplevelArtifacts(
      Environment env, ExtendedEventHandler.Postable postable, ActionInputMap inputMap)
      throws CompletionFunctionException, InterruptedException {
    // For skymeld, a non-toplevel target might become a toplevel after it has been executed. This
    // is the last chance to download the missing toplevel outputs in this case before sending out
    // TargetCompleteEvent. See https://github.com/bazelbuild/bazel/issues/20737.
    if (!isSkymeld.get()) {
      return;
    }

    var outputService = skyframeActionExecutor.getOutputService();
    if (outputService == null) {
      return;
    }

    var actionInputPrefetcher = skyframeActionExecutor.getActionInputPrefetcher();
    if (actionInputPrefetcher == null || actionInputPrefetcher == ActionInputPrefetcher.NONE) {
      return;
    }

    var remoteArtifactChecker = outputService.getRemoteArtifactChecker();
    if (remoteArtifactChecker == RemoteArtifactChecker.TRUST_ALL) {
      return;
    }

    ImmutableMap<String, ArtifactsInOutputGroup> allOutputGroups;
    Runfiles runfiles = null;
    if (postable instanceof TargetCompleteEvent targetCompleteEvent) {
      allOutputGroups = targetCompleteEvent.getOutputs();
      runfiles = targetCompleteEvent.getExecutableTargetData().getRunfiles();
    } else if (postable instanceof AspectCompleteEvent aspectCompleteEvent) {
      allOutputGroups = aspectCompleteEvent.getOutputs();
    } else {
      return;
    }

    var futures = new ArrayList<ListenableFuture<Void>>();
    for (var outputGroup : allOutputGroups.values()) {
      if (!outputGroup.areImportant()) {
        continue;
      }

      for (var artifact : outputGroup.getArtifacts().toList()) {
        downloadArtifact(
            env, remoteArtifactChecker, actionInputPrefetcher, inputMap, artifact, futures);
      }
    }

    if (runfiles != null) {
      for (var artifact : runfiles.getAllArtifacts().toList()) {
        downloadArtifact(
            env, remoteArtifactChecker, actionInputPrefetcher, inputMap, artifact, futures);
      }
    }

    try {
      var unused = Futures.whenAllSucceed(futures).call(() -> null, directExecutor()).get();
    } catch (ExecutionException e) {
      var cause = e.getCause();
      if (cause instanceof ActionExecutionException aee) {
        throw new CompletionFunctionException(aee);
      }
      throw new RuntimeException(cause);
    }
  }

  private void downloadArtifact(
      Environment env,
      RemoteArtifactChecker remoteArtifactChecker,
      ActionInputPrefetcher actionInputPrefetcher,
      ActionInputMap inputMap,
      Artifact artifact,
      List<ListenableFuture<Void>> futures
  ) throws InterruptedException {
    if (!(artifact instanceof DerivedArtifact derivedArtifact)) {
      return;
    }

    // Metadata can be null during error bubbling, only download outputs that are already
    // generated. b/342188273
    if (artifact.isTreeArtifact()) {
      var treeMetadata = inputMap.getTreeMetadata(artifact.getExecPath());
      if (treeMetadata == null) {
        return;
      }

      var filesToDownload = new ArrayList<ActionInput>(treeMetadata.getChildValues().size());
      for (var child : treeMetadata.getChildValues().entrySet()) {
        var treeFile = child.getKey();
        var metadata = child.getValue();
        if (metadata.isRemote()
            && !remoteArtifactChecker.shouldTrustRemoteArtifact(
            treeFile, (RemoteFileArtifactValue) metadata)) {
          filesToDownload.add(treeFile);
        }
      }
      if (!filesToDownload.isEmpty()) {
        var action =
            ActionUtils.getActionForLookupData(env, derivedArtifact.getGeneratingActionKey());
        var future =
            actionInputPrefetcher.prefetchFiles(
                action, filesToDownload, inputMap::getInputMetadata, Priority.LOW);
        futures.add(
            Futures.catchingAsync(
                future,
                Throwable.class,
                e ->
                    Futures.immediateFailedFuture(
                        new ActionExecutionException(
                            e,
                            action,
                            true,
                            DetailedExitCode.of(
                                FailureDetail.newBuilder().setMessage(e.getMessage()).build()))),
                directExecutor()));
      }
    } else {
      var metadata = inputMap.getInputMetadata(artifact);
      if (metadata == null) {
        return;
      }

      if (metadata.isRemote()
          && !remoteArtifactChecker.shouldTrustRemoteArtifact(
          artifact, (RemoteFileArtifactValue) metadata)) {
        var action =
            ActionUtils.getActionForLookupData(env, derivedArtifact.getGeneratingActionKey());
        var future =
            actionInputPrefetcher.prefetchFiles(
                action, ImmutableList.of(artifact), inputMap::getInputMetadata, Priority.LOW);
        futures.add(
            Futures.catchingAsync(
                future,
                Throwable.class,
                e ->
                    Futures.immediateFailedFuture(
                        new ActionExecutionException(
                            e,
                            action,
                            true,
                            DetailedExitCode.of(
                                FailureDetail.newBuilder()
                                    .setMessage(e.getMessage())
                                    .setRemoteExecution(
                                        RemoteExecution.newBuilder()
                                            .setCode(
                                                RemoteExecution.Code
                                                    .TOPLEVEL_OUTPUTS_DOWNLOAD_FAILURE)
                                            .build())
                                    .build()))),
                directExecutor()));
      }
    }
  }

  private void handleSourceFileError(
      Artifact input,
      DetailedExitCode detailedExitCode,
      NestedSetBuilder<Cause> rootCausesBuilder,
      Environment env,
      ValueT value,
      KeyT key)
      throws InterruptedException {
    LabelCause cause =
        ActionExecutionFunction.createLabelCause(
            input, detailedExitCode, key.actionLookupKey().getLabel(), bugReporter);
    rootCausesBuilder.add(cause);
    env.getListener().handle(completor.getRootCauseError(value, key, cause, env));
    skyframeActionExecutor.recordExecutionError();
  }

  @Nullable
  static <ValueT extends ConfiguredObjectValue>
      Pair<ValueT, ArtifactsToBuild> getValueAndArtifactsToBuild(
          TopLevelActionLookupKeyWrapper key, Environment env) throws InterruptedException {
    @SuppressWarnings("unchecked")
    ValueT value = (ValueT) env.getValue(key.actionLookupKey());
    if (env.valuesMissing()) {
      return null;
    }

    TopLevelArtifactContext topLevelContext = key.topLevelArtifactContext();
    ArtifactsToBuild artifactsToBuild =
        TopLevelArtifactHelper.getAllArtifactsToBuild(value.getConfiguredObject(), topLevelContext);
    return Pair.of(value, artifactsToBuild);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((TopLevelActionLookupKeyWrapper) skyKey).actionLookupKey().getLabel());
  }

  private static final class CompletionFunctionException extends SkyFunctionException {
    private final ActionExecutionException actionException;

    CompletionFunctionException(ActionExecutionException e) {
      super(e, Transience.PERSISTENT);
      this.actionException = e;
    }

    CompletionFunctionException(InputFileErrorException e) {
      // Not transient from the point of view of this SkyFunction.
      super(e, Transience.PERSISTENT);
      this.actionException = null;
    }

    @Override
    public boolean isCatastrophic() {
      return actionException != null && actionException.isCatastrophe();
    }
  }
}
