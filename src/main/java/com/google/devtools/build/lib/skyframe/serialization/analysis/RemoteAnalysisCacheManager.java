// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.Strings.nullToEmpty;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.PathFragmentPrefixTrie;
import com.google.devtools.build.lib.collect.PathFragmentPrefixTrie.PathFragmentPrefixTrieException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.pkgcache.PackagePathCodecDependencies;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InstrumentationOutput;
import com.google.devtools.build.lib.runtime.InstrumentationOutputFactory.DestinationRelativeTo;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching;
import com.google.devtools.build.lib.skyframe.PrerequisitePackageFunction;
import com.google.devtools.build.lib.skyframe.SkyfocusOptions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.WorkspaceInfoFromDiff;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.lib.skyframe.serialization.SkycacheMetadataParams;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.LongVersionClientId;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheClient.LookupTopLevelTargetsResult;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions.RemoteAnalysisCacheMode;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Root.RootCodecDependencies;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** A collection of dependencies and minor bits of functionality for remote analysis caching. */
// Non-final for mockability
public class RemoteAnalysisCacheManager implements RemoteAnalysisCachingDependenciesProvider {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final RemoteAnalysisCacheMode mode;

  private final Future<? extends RemoteAnalysisCacheClient> analysisCacheClient;
  private final Future<? extends AnalysisCacheInvalidator> analysisCacheInvalidator;

  private final Collection<Label> topLevelTargets;
  private final Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher;

  private final ExtendedEventHandler eventHandler;

  private final boolean areMetadataQueriesEnabled;
  private final SkycacheMetadataParams skycacheMetadataParams;

  private boolean bailedOut;

  private final boolean minimizeMemory;

  private static final long CLIENT_LOOKUP_TIMEOUT_SEC = 20L;

  private static <T> T resolveWithTimeout(Future<? extends T> future, String what)
      throws InterruptedException {
    if (future == null) {
      return null;
    }
    try {
      return future.get(CLIENT_LOOKUP_TIMEOUT_SEC, SECONDS);
    } catch (ExecutionException | TimeoutException e) {
      logger.atWarning().withCause(e).log("Unable to initialize %s", what);
      return null;
    }
  }

  /**
   * A collection of various parts of this class that various parts of Bazel (cache reading, cache
   * writing, in-memory bookkeeping) need.
   *
   * <p>This record will eventually go away; the reason why they can't yet is that {@code
   * #forAnalysis} needs to be able to return a {@link DisabledDependenciesProvider}.
   */
  public record AnalysisDeps(
      RemoteAnalysisCachingDependenciesProvider deps,
      RemoteAnalysisCacheReaderDepsProvider readerDeps,
      SerializationDependenciesProvider serializationDeps) {}

  public static AnalysisDeps forAnalysis(
      CommandEnvironment env,
      Optional<PathFragmentPrefixTrie> maybeActiveDirectoriesMatcher,
      Collection<Label> topLevelTargets,
      BuildOptions topLevelOptions,
      Map<String, String> userOptions,
      Set<String> projectSclOptions)
      throws InterruptedException, AbruptExitException, InvalidConfigurationException {
    // Bail out early if needed
    var options = env.getOptions().getOptions(RemoteAnalysisCachingOptions.class);
    if (options == null
        || !env.getCommand().buildPhase().executes()
        || options.mode == RemoteAnalysisCacheMode.OFF) {
      return new AnalysisDeps(
          DisabledDependenciesProvider.INSTANCE,
          DisabledDependenciesProvider.INSTANCE,
          DisabledDependenciesProvider.INSTANCE);
    }

    // Set up active directory matcher

    Optional<PathFragmentPrefixTrie> maybeActiveDirectoriesMatcherFromFlags =
        finalizeActiveDirectoriesMatcher(env, maybeActiveDirectoriesMatcher, options.mode);
    Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher =
        maybeActiveDirectoriesMatcherFromFlags.map(v -> pi -> v.includes(pi.getPackageFragment()));

    // Compute versions we are evaluating at

    var workspaceInfoFromDiff = env.getWorkspaceInfoFromDiff();
    if (workspaceInfoFromDiff == null) {
      workspaceInfoFromDiff = new WorkspaceInfoFromDiff() {}; // Rely on default implementations
    }
    ClientId clientId =
        workspaceInfoFromDiff
            .getSnapshot()
            .orElse(new LongVersionClientId(workspaceInfoFromDiff.getEvaluatingVersion().getVal()));
    HashCode blazeInstallMD5 = computeBlazeInstallMD5(env, options);

    FrontierNodeVersion frontierNodeVersion =
        new FrontierNodeVersion(
            topLevelOptions.checksum(),
            blazeInstallMD5,
            workspaceInfoFromDiff.getEvaluatingVersion(),
            nullToEmpty(options.analysisCacheKeyDistinguisherForTesting),
            env.getUseFakeStampData(),
            workspaceInfoFromDiff.getSnapshot());
    env.getRemoteAnalysisCachingEventListener().recordSkyValueVersion(frontierNodeVersion);
    env.getRemoteAnalysisCachingEventListener().setClientId(clientId);
    logger.atInfo().log(
        "Remote analysis caching SkyValue version: %s (actual evaluating version: %s)",
        frontierNodeVersion, workspaceInfoFromDiff.getEvaluatingVersion());

    // Create various objets we need

    RemoteAnalysisJsonLogWriter jsonLogWriter = createJsonLogWriterMaybe(env, options);
    ListenableFuture<ObjectCodecs> objectCodecs = createObjectCodecs(env);

    RemoteAnalysisCachingServicesSupplier servicesSupplier =
        env.getBlazeWorkspace().remoteAnalysisCachingServicesSupplier();
    servicesSupplier.configure(options, clientId, env.getCommandId().toString(), jsonLogWriter);

    // Set up parameters for the metadata store, if needed

    SkycacheMetadataParams skycacheMetadataParams = servicesSupplier.getSkycacheMetadataParams();
    boolean areMetadataQueriesEnabled =
        skycacheMetadataParams != null && options.analysisCacheEnableMetadataQueries;

    if (areMetadataQueriesEnabled) {
      skycacheMetadataParams.init(
          workspaceInfoFromDiff.getEvaluatingVersion().getVal(),
          String.format("%s-%s", BlazeVersionInfo.instance().getReleaseName(), blazeInstallMD5),
          topLevelTargets.stream().map(Label::toString).collect(toImmutableList()),
          env.getUseFakeStampData(),
          userOptions,
          projectSclOptions);
    }

    if (skycacheMetadataParams != null) {
      skycacheMetadataParams.setConfigurationHash(topLevelOptions.checksum());
      skycacheMetadataParams.setOriginalConfigurationOptions(
          getConfigurationOptionsAsStrings(topLevelOptions));
    }

    // Create the return values

    var deps =
        new RemoteAnalysisCacheDeps(
            env.getReporter(),
            options.mode,
            options.analysisCacheBailOnMissingFingerprint,
            servicesSupplier,
            env.getRemoteAnalysisCachingEventListener(),
            jsonLogWriter,
            objectCodecs,
            frontierNodeVersion,
            activeDirectoriesMatcher,
            options.serializedFrontierProfile);

    ListenableFuture<AnalysisCacheInvalidator> analysisCacheInvalidator =
        createAnalysisCacheInvalidator(
            env.getReporter(),
            clientId,
            frontierNodeVersion,
            objectCodecs,
            servicesSupplier.getFingerprintValueService(),
            servicesSupplier.getAnalysisCacheClient());

    var manager =
        new RemoteAnalysisCacheManager(
            options.mode,
            areMetadataQueriesEnabled,
            env.getReporter(),
            skycacheMetadataParams,
            servicesSupplier.getAnalysisCacheClient(),
            analysisCacheInvalidator,
            topLevelTargets,
            activeDirectoriesMatcher,
            options.skycacheMinimizeMemory);

    // Bail out if needed

    return switch (options.mode) {
      case RemoteAnalysisCacheMode.DUMP_UPLOAD_MANIFEST_ONLY, RemoteAnalysisCacheMode.UPLOAD ->
          new AnalysisDeps(manager, deps, deps);
      case RemoteAnalysisCacheMode.DOWNLOAD -> {
        if (deps.getAnalysisCacheClient() == null) {
          if (Strings.isNullOrEmpty(options.analysisCacheService)) {
            env.getReporter()
                .handle(
                    Event.warn(
                        "--experimental_remote_analysis_cache_mode=DOWNLOAD was requested but"
                            + " --experimental_analysis_cache_service was not specified. Falling"
                            + " back on local evaluation."));
          } else {
            env.getReporter()
                .handle(
                    Event.warn(
                        "Failed to establish connection to AnalysisCacheService. Falling back to"
                            + " on local evaluation."));
          }
          yield new AnalysisDeps(
              DisabledDependenciesProvider.INSTANCE,
              DisabledDependenciesProvider.INSTANCE,
              DisabledDependenciesProvider.INSTANCE);
        }
        yield new AnalysisDeps(manager, deps, deps);
      }
      default ->
          throw new IllegalStateException("Unknown RemoteAnalysisCacheMode: " + options.mode);
    };
  }

  /**
   * Determines the active directories matcher for remote analysis caching operations.
   *
   * <p>For upload mode, optionally check the --experimental_active_directories flag if the project
   * file matcher is not present.
   *
   * <p>For download mode, the matcher is always empty.
   */
  private static Optional<PathFragmentPrefixTrie> finalizeActiveDirectoriesMatcher(
      CommandEnvironment env,
      Optional<PathFragmentPrefixTrie> maybeProjectFileMatcher,
      RemoteAnalysisCacheMode mode)
      throws InvalidConfigurationException {
    return switch (mode) {
      case DOWNLOAD, OFF -> Optional.empty();
      case UPLOAD, DUMP_UPLOAD_MANIFEST_ONLY -> {
        // Upload or Dump mode: allow overriding the project file matcher with the active
        // directories flag.
        List<String> activeDirectoriesFromFlag =
            env.getOptions().getOptions(SkyfocusOptions.class).activeDirectories;
        var result = maybeProjectFileMatcher;
        if (!activeDirectoriesFromFlag.isEmpty()) {
          env.getReporter()
              .handle(
                  Event.warn(
                      "Specifying --experimental_active_directories will override the active"
                          + " directories specified in the PROJECT.scl file"));
          try {
            result = Optional.of(PathFragmentPrefixTrie.of(activeDirectoriesFromFlag));
          } catch (PathFragmentPrefixTrieException e) {
            throw new InvalidConfigurationException(
                "Active directories configuration error: " + e.getMessage(), Code.INVALID_PROJECT);
          }
        }

        if (result.isEmpty() || !result.get().hasIncludedPaths()) {
          env.getReporter()
              .handle(
                  Event.warn(
                      "No active directories were found. Falling back on full serialization."));
          yield Optional.empty();
        }
        yield result;
      }
    };
  }

  private RemoteAnalysisCacheManager(
      RemoteAnalysisCacheMode mode,
      boolean areMetadataQueriesEnabled,
      ExtendedEventHandler eventHandler,
      SkycacheMetadataParams skycacheMetadataParams,
      Future<? extends RemoteAnalysisCacheClient> analysisCacheClient,
      Future<? extends AnalysisCacheInvalidator> analysisCacheInvalidator,
      Collection<Label> topLevelTargets,
      Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher,
      boolean minimizeMemory) {
    this.mode = mode;
    this.analysisCacheClient = analysisCacheClient;
    this.analysisCacheInvalidator = analysisCacheInvalidator;
    this.topLevelTargets = topLevelTargets;
    this.activeDirectoriesMatcher = activeDirectoriesMatcher;
    this.minimizeMemory = minimizeMemory;
    this.eventHandler = eventHandler;
    this.skycacheMetadataParams = skycacheMetadataParams;
    this.areMetadataQueriesEnabled = areMetadataQueriesEnabled;
  }

  private static ObjectCodecs initAnalysisObjectCodecs(
      ObjectCodecRegistry registry,
      RuleClassProvider ruleClassProvider,
      SkyframeExecutor skyframeExecutor,
      BlazeDirectories directories) {
    var roots = ImmutableList.<Root>builder().add(Root.fromPath(directories.getWorkspace()));
    // TODO: b/406458763 - clean this up
    if (Ascii.equalsIgnoreCase(directories.getProductName(), "blaze")) {
      roots.add(Root.fromPath(directories.getBlazeExecRoot()));
    }

    ImmutableClassToInstanceMap.Builder<Object> serializationDeps =
        ImmutableClassToInstanceMap.builder()
            .put(
                ArtifactSerializationContext.class,
                skyframeExecutor.getSkyframeBuildView().getArtifactFactory()::getSourceArtifact)
            .put(RuleClassProvider.class, ruleClassProvider)
            .put(RootCodecDependencies.class, new RootCodecDependencies(roots.build()))
            .put(PackagePathCodecDependencies.class, skyframeExecutor::getPackagePathEntries)
            // This is needed to determine TargetData for a ConfiguredTarget during serialization.
            .put(PrerequisitePackageFunction.class, skyframeExecutor::getExistingPackage);

    return new ObjectCodecs(registry, serializationDeps.build());
  }

  private static ListenableFuture<ObjectCodecs> createObjectCodecs(CommandEnvironment env) {
    return Futures.submit(
        () ->
            initAnalysisObjectCodecs(
                requireNonNull(env.getBlazeWorkspace().getAnalysisObjectCodecRegistrySupplier())
                    .get(),
                env.getRuntime().getRuleClassProvider(),
                env.getBlazeWorkspace().getSkyframeExecutor(),
                env.getDirectories()),
        commonPool());
  }

  @Nullable // In case we don't expect a connection to the analysis cache server
  private static ListenableFuture<AnalysisCacheInvalidator> createAnalysisCacheInvalidator(
      ExtendedEventHandler eventHandler,
      ClientId clientId,
      FrontierNodeVersion frontierNodeVersion,
      ListenableFuture<? extends ObjectCodecs> objectCodecs,
      ListenableFuture<? extends FingerprintValueService> fingerprintValueService,
      ListenableFuture<? extends RemoteAnalysisCacheClient> analysisCacheClient) {
    if (analysisCacheClient == null) {
      return immediateFuture(null);
    }
    return Futures.whenAllSucceed(objectCodecs, fingerprintValueService, analysisCacheClient)
        .call(
            () ->
                new AnalysisCacheInvalidator(
                    analysisCacheClient.get(),
                    objectCodecs.get(),
                    fingerprintValueService.get(),
                    frontierNodeVersion,
                    clientId,
                    eventHandler),
            commonPool());
  }

  private static HashCode computeBlazeInstallMD5(
      CommandEnvironment env, RemoteAnalysisCachingOptions options) throws AbruptExitException {
    if (options.serverChecksumOverride == null) {
      return requireNonNull(env.getDirectories().getInstallMD5());
    }

    if (options.mode != RemoteAnalysisCacheMode.DOWNLOAD) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage("Server checksum override can only be used in download mode")
                  .setRemoteAnalysisCaching(
                      RemoteAnalysisCaching.newBuilder()
                          .setCode(RemoteAnalysisCaching.Code.INCOMPATIBLE_OPTIONS))
                  .build()));
    }

    env.getReporter()
        .handle(
            Event.warn(
                String.format(
                    "Skycache will use server checksum '%s' instead of '%s', which describes"
                        + " this binary. This may cause crashes or even silent incorrectness."
                        + " You've been warned! (check the documentation of the command line "
                        + " flag for more details)",
                    options.serverChecksumOverride, env.getDirectories().getInstallMD5())));

    return options.serverChecksumOverride;
  }

  @Nullable
  private static RemoteAnalysisJsonLogWriter createJsonLogWriterMaybe(
      CommandEnvironment env, RemoteAnalysisCachingOptions options) throws AbruptExitException {
    if (options.jsonLog == null) {
      return null;
    }
    try {
      InstrumentationOutput jsonLog =
          env.getRuntime()
              .getInstrumentationOutputFactory()
              .createInstrumentationOutput(
                  "remote_cache_jsonlog",
                  PathFragment.create(options.jsonLog),
                  DestinationRelativeTo.WORKING_DIRECTORY_OR_HOME,
                  env,
                  env.getReporter(),
                  /* append= */ false,
                  /* internal= */ false);
      var result =
          new RemoteAnalysisJsonLogWriter(
              new JsonWriter(
                  new OutputStreamWriter(
                      new BufferedOutputStream(jsonLog.createOutputStream(), 262144), ISO_8859_1)));
      env.getReporter()
          .handle(Event.info(String.format("Writing Skycache JSON log to '%s'", options.jsonLog)));
      return result;
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "Cannot open remote analysis JSON log file '%s': %s",
                          options.jsonLog, e.getMessage()))
                  .setRemoteAnalysisCaching(
                      RemoteAnalysisCaching.newBuilder()
                          .setCode(RemoteAnalysisCaching.Code.CANNOT_OPEN_LOG_FILE))
                  .build()));
    }
  }

  @Override
  public RemoteAnalysisCacheMode mode() {
    return mode;
  }

  @Override
  public void queryMetadataAndMaybeBailout() throws InterruptedException {
    Preconditions.checkState(mode == RemoteAnalysisCacheMode.DOWNLOAD);
    if (!areMetadataQueriesEnabled) {
      return;
    }
    if (skycacheMetadataParams.getTargets().isEmpty()) {
      eventHandler.handle(
          Event.warn("Skycache: Not querying Skycache metadata because invocation has no targets"));
    } else {
      try {
        LookupTopLevelTargetsResult result =
            resolveWithTimeout(analysisCacheClient, "analysis cache client")
                .lookupTopLevelTargets(
                    skycacheMetadataParams.getEvaluatingVersion(),
                    skycacheMetadataParams.getConfigurationHash(),
                    skycacheMetadataParams.getUseFakeStampData(),
                    skycacheMetadataParams.getBazelVersion());

        Event event =
            switch (result.status()) {
              case MATCH_STATUS_MATCH -> Event.info("Skycache: " + result.statusMessage());
              case MATCH_STATUS_FAILURE -> Event.warn("Skycache: " + result.statusMessage());
              default -> {
                bailedOut = true;
                yield Event.warn("Skycache: " + result.statusMessage());
              }
            };
        eventHandler.handle(event);
      } catch (ExecutionException | TimeoutException e) {
        eventHandler.handle(Event.warn("Skycache: Error with metadata store: " + e.getMessage()));
      }
    }
  }

  /**
   * This method returns all the configuration affecting options regardless of where they have been
   * set and regardless of whether they have been set at all (using default values).
   *
   * <p>They exclude test options since test options do not affect the configuration checksum used
   * by Skycache's frontier node versions. Test configuration trimming is an optimization that
   * removes test options from all but *_test targets. The details of the optimization aren't
   * relevant here for this method, the only relevant part is that the top level target checksum is
   * always computed without test options.
   */
  private static ImmutableSet<String> getConfigurationOptionsAsStrings(BuildOptions targetOptions) {
    ImmutableSet.Builder<String> allOptionsAsStringsBuilder = new ImmutableSet.Builder<>();

    // Collect a list of BuildOptions, excluding TestOptions.
    targetOptions.getStarlarkOptions().keySet().stream()
        .map(Object::toString)
        .forEach(allOptionsAsStringsBuilder::add);
    for (FragmentOptions fragmentOptions : targetOptions.getNativeOptions()) {
      if (fragmentOptions.getClass().equals(TestConfiguration.TestOptions.class)) {
        continue;
      }
      fragmentOptions.asMap().keySet().forEach(allOptionsAsStringsBuilder::add);
    }
    return allOptionsAsStringsBuilder.build();
  }

  @Override
  public Set<SkyKey> lookupKeysToInvalidate(
      ImmutableSet<SkyKey> keysToLookup,
      RemoteAnalysisCachingServerState remoteAnalysisCachingState)
      throws InterruptedException {
    AnalysisCacheInvalidator invalidator =
        resolveWithTimeout(analysisCacheInvalidator, "analysis cache invalidator");
    if (invalidator == null) {
      // We need to know which keys to invalidate but we don't have an invalidator, presumably
      // because the backend services couldn't be contacted. Play if safe and invalidate every
      // value retrieved from the remote cache.
      return keysToLookup;
    }
    return invalidator.lookupKeysToInvalidate(keysToLookup, remoteAnalysisCachingState);
  }

  @Override
  public boolean bailedOut() {
    return bailedOut;
  }

  @Override
  public void computeSelectionAndMinimizeMemory(InMemoryGraph graph) {
    FrontierSerializer.computeSelectionAndMinimizeMemory(
        graph, topLevelTargets, activeDirectoriesMatcher);
  }

  @Override
  public boolean shouldMinimizeMemory() {
    return minimizeMemory;
  }

  private static class RemoteAnalysisCacheDeps
      implements SerializationDependenciesProvider, RemoteAnalysisCacheReaderDepsProvider {
    private final RemoteAnalysisCacheMode mode;
    private final boolean bailOutOnMissingFingerprint;
    private final String serializedFrontierProfile;
    private final Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher;
    private final RemoteAnalysisCachingEventListener listener;
    private final FrontierNodeVersion frontierNodeVersion;
    @Nullable private final RemoteAnalysisJsonLogWriter jsonLogWriter;

    private final ListenableFuture<ObjectCodecs> objectCodecs;
    private final ListenableFuture<FingerprintValueService> fingerprintValueServiceFuture;
    @Nullable
    private final ListenableFuture<? extends RemoteAnalysisCacheClient> analysisCacheClient;
    @Nullable private final ListenableFuture<? extends RemoteAnalysisMetadataWriter> metadataWriter;

    private final AtomicBoolean bailedOut = new AtomicBoolean();
    private final ExtendedEventHandler eventHandler;

    RemoteAnalysisCacheDeps(
        ExtendedEventHandler eventHandler,
        RemoteAnalysisCacheMode mode,
        boolean bailOutOnMissingFingerprint,
        RemoteAnalysisCachingServicesSupplier servicesSupplier,
        RemoteAnalysisCachingEventListener listener,
        RemoteAnalysisJsonLogWriter jsonLogWriter,
        ListenableFuture<ObjectCodecs> objectCodecs,
        FrontierNodeVersion frontierNodeVersion,
        Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher,
        String serializedFrontierProfile) {
      this.mode = mode;
      this.bailOutOnMissingFingerprint = bailOutOnMissingFingerprint;
      this.serializedFrontierProfile = serializedFrontierProfile;
      this.activeDirectoriesMatcher = activeDirectoriesMatcher;
      this.eventHandler = eventHandler;

      this.jsonLogWriter = jsonLogWriter;

      this.objectCodecs = objectCodecs;
      this.listener = listener;

      this.frontierNodeVersion = frontierNodeVersion;

      this.fingerprintValueServiceFuture = servicesSupplier.getFingerprintValueService();
      this.metadataWriter = servicesSupplier.getMetadataWriter();
      this.analysisCacheClient = servicesSupplier.getAnalysisCacheClient();
    }

    @Override
    public RemoteAnalysisCacheMode mode() {
      return mode;
    }

    @Override
    public String getSerializedFrontierProfile() {
      return serializedFrontierProfile;
    }

    @Override
    public Optional<Predicate<PackageIdentifier>> getActiveDirectoriesMatcher() {
      return activeDirectoriesMatcher;
    }

    @Override
    public FrontierNodeVersion getSkyValueVersion() throws InterruptedException {
      return frontierNodeVersion;
    }

    @Override
    public ObjectCodecs getObjectCodecs() throws InterruptedException {
      try {
        return objectCodecs.get();
      } catch (ExecutionException e) {
        throw new IllegalStateException("Failed to initialize ObjectCodecs", e);
      }
    }

    @Override
    public FingerprintValueService getFingerprintValueService() throws InterruptedException {
      return resolveWithTimeout(fingerprintValueServiceFuture, "fingerprint value service");
    }

    @Override
    public KeyValueWriter getFileInvalidationWriter() throws InterruptedException {
      return getFingerprintValueService();
    }

    @Override
    @Nullable
    public RemoteAnalysisCacheClient getAnalysisCacheClient() throws InterruptedException {
      return resolveWithTimeout(analysisCacheClient, "analysis cache client");
    }

    @Override
    @Nullable
    public RemoteAnalysisMetadataWriter getMetadataWriter() throws InterruptedException {
      return resolveWithTimeout(metadataWriter, "metadata writer");
    }

    @Nullable
    @Override
    public RemoteAnalysisJsonLogWriter getJsonLogWriter() {
      return jsonLogWriter;
    }

    @Override
    public void recordRetrievalResult(RetrievalResult retrievalResult, SkyKey key) {
      listener.recordRetrievalResult(retrievalResult, key);
    }

    @Override
    public void recordSerializationException(SerializationException e, SkyKey key) {
      listener.recordSerializationException(e, key);
    }

    @Override
    public boolean shouldBailOutOnMissingFingerprint() {
      if (!bailOutOnMissingFingerprint) {
        return false;
      }
      if (bailedOut.get()) {
        return true;
      }

      try {
        FingerprintValueService service = getFingerprintValueService();
        boolean retVal = service != null && service.getStats().entriesNotFound() > 0;
        if (retVal) {
          bailedOut.set(true);
          eventHandler.handle(
              Event.warn(
                  "Skycache: falling back to local evaluation due to unexpected missing cache"
                      + " entries"));
          analysisCacheClient.get().bailOutDueToMissingFingerprint();
        }
        return retVal;
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return false;
      } catch (ExecutionException e) {
        throw new IllegalStateException(
            "At this point the Skycache client should have been initialized", e);
      }
    }
  }
}
