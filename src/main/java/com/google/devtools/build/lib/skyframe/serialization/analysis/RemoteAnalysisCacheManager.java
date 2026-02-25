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
import com.google.devtools.build.lib.analysis.BuildView;
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
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InstrumentationOutput;
import com.google.devtools.build.lib.runtime.InstrumentationOutputFactory.DestinationRelativeTo;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching;
import com.google.devtools.build.lib.skyframe.PrerequisitePackageFunction;
import com.google.devtools.build.lib.skyframe.SkyfocusOptions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
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
import com.google.devtools.build.skyframe.IntVersion;
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
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** A collection of dependencies and minor bits of functionality for remote analysis caching. */
// Non-final for mockability
public class RemoteAnalysisCacheManager implements RemoteAnalysisCachingDependenciesProvider {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final RemoteAnalysisCacheMode mode;

  private final Future<? extends RemoteAnalysisCacheClient> analysisCacheClient;
  private final RemoteAnalysisCacheDeps deps;

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
      Collection<Label> targets,
      Map<String, String> userOptions,
      Set<String> projectSclOptions)
      throws InterruptedException, AbruptExitException, InvalidConfigurationException {
    var options = env.getOptions().getOptions(RemoteAnalysisCachingOptions.class);
    if (options == null
        || !env.getCommand().buildPhase().executes()
        || options.mode == RemoteAnalysisCacheMode.OFF) {
      return new AnalysisDeps(
          DisabledDependenciesProvider.INSTANCE,
          DisabledDependenciesProvider.INSTANCE,
          DisabledDependenciesProvider.INSTANCE);
    }

    Optional<PathFragmentPrefixTrie> maybeActiveDirectoriesMatcherFromFlags =
        finalizeActiveDirectoriesMatcher(env, maybeActiveDirectoriesMatcher, options.mode);
    Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher =
        maybeActiveDirectoriesMatcherFromFlags.map(v -> pi -> v.includes(pi.getPackageFragment()));
    RemoteAnalysisJsonLogWriter jsonLogWriter = createJsonLogWriterMaybe(env, options);
    Future<ObjectCodecs> objectCodecs = createObjectCodecs(env);

    var deps =
        new RemoteAnalysisCacheDeps(
            env,
            jsonLogWriter,
            objectCodecs,
            options,
            activeDirectoriesMatcher,
            options.serializedFrontierProfile,
            targets);

    var manager =
        new RemoteAnalysisCacheManager(
            env.getReporter(),
            env.getBlazeWorkspace()
                .remoteAnalysisCachingServicesSupplier()
                .getSkycacheMetadataParams(),
            options,
            env.getBlazeWorkspace()
                .remoteAnalysisCachingServicesSupplier()
                .getAnalysisCacheClient(),
            deps,
            env.getUseFakeStampData(),
            options.skycacheMinimizeMemory,
            userOptions,
            projectSclOptions);

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
      ExtendedEventHandler eventHandler,
      SkycacheMetadataParams skycacheMetadataParams,
      RemoteAnalysisCachingOptions options,
      Future<? extends RemoteAnalysisCacheClient> analysisCacheClient,
      RemoteAnalysisCacheDeps deps,
      boolean useFakeStampData,
      boolean minimizeMemory,
      Map<String, String> userOptions,
      Set<String> projectSclOptions) {
    this.mode = options.mode;
    this.analysisCacheClient = analysisCacheClient;
    this.deps = deps;
    this.minimizeMemory = minimizeMemory;
    this.eventHandler = eventHandler;
    this.skycacheMetadataParams = skycacheMetadataParams;
    this.areMetadataQueriesEnabled =
        skycacheMetadataParams != null && options.analysisCacheEnableMetadataQueries;
    if (areMetadataQueriesEnabled) {
      this.skycacheMetadataParams.init(
          deps.getEvaluatingVersion().getVal(),
          String.format(
              "%s-%s", BlazeVersionInfo.instance().getReleaseName(), deps.getBlazeInstallMD5()),
          deps.getTopLevelTargets().stream().map(Label::toString).collect(toImmutableList()),
          useFakeStampData,
          userOptions,
          projectSclOptions);
    }
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
  public void setTopLevelBuildOptions(BuildOptions buildOptions) {
    deps.setTopLevelBuildOptions(buildOptions);
    if (skycacheMetadataParams != null) {
      skycacheMetadataParams.setConfigurationHash(buildOptions.checksum());
      skycacheMetadataParams.setOriginalConfigurationOptions(
          getConfigurationOptionsAsStrings(buildOptions));
    }
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
    AnalysisCacheInvalidator invalidator = deps.getAnalysisCacheInvalidator();
    if (invalidator == null) {
      return ImmutableSet.of();
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
        graph, deps.getTopLevelTargets(), deps.getActiveDirectoriesMatcher());
  }

  @Override
  public boolean shouldMinimizeMemory() {
    return minimizeMemory;
  }

  private static class RemoteAnalysisCacheDeps
      implements SerializationDependenciesProvider, RemoteAnalysisCacheReaderDepsProvider {
    private final RemoteAnalysisCacheMode mode;
    private final String serializedFrontierProfile;
    private final Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher;
    private final RemoteAnalysisCachingEventListener listener;
    private final HashCode blazeInstallMD5;
    @Nullable private final String distinguisher;
    private final boolean useFakeStampData;

    /** Cache lookup parameter requiring integration with external version control. */
    private final IntVersion evaluatingVersion;

    /** Cache lookup parameter requiring integration with external version control. */
    private final Optional<ClientId> snapshot;

    @Nullable private final RemoteAnalysisJsonLogWriter jsonLogWriter;

    private final Future<ObjectCodecs> objectCodecs;
    private final Future<FingerprintValueService> fingerprintValueServiceFuture;
    @Nullable private final Future<? extends RemoteAnalysisCacheClient> analysisCacheClient;
    @Nullable private final Future<? extends RemoteAnalysisMetadataWriter> metadataWriter;
    @Nullable private volatile AnalysisCacheInvalidator analysisCacheInvalidator;

    private final ExtendedEventHandler eventHandler;

    // Non-final because the top level BuildConfigurationValue is determined just before analysis
    // begins in BuildView for the download/deserialization pass, which is later than when this
    // object was created in BuildTool.
    private BuildOptions topLevelBuildOptions;

    private final Collection<Label> topLevelTargets;

    private volatile FrontierNodeVersion frontierNodeVersionSingleton = null;

    RemoteAnalysisCacheDeps(
        CommandEnvironment env,
        RemoteAnalysisJsonLogWriter jsonLogWriter,
        Future<ObjectCodecs> objectCodecs,
        RemoteAnalysisCachingOptions options,
        Optional<Predicate<PackageIdentifier>> activeDirectoriesMatcher,
        String serializedFrontierProfile,
        Collection<Label> targets)
        throws AbruptExitException, InterruptedException {
      this.topLevelTargets = targets;
      this.mode = options.mode;
      this.serializedFrontierProfile = serializedFrontierProfile;
      this.activeDirectoriesMatcher = activeDirectoriesMatcher;
      this.eventHandler = env.getReporter();

      this.jsonLogWriter = jsonLogWriter;

      this.objectCodecs = objectCodecs;
      this.listener = env.getRemoteAnalysisCachingEventListener();
      if (env.getSkyframeBuildView().getBuildConfiguration() != null) {
        // null at construction time during deserializing build. Will be set later during analysis.
        this.topLevelBuildOptions =
            BuildView.getTopLevelConfigurationTrimmedOfTestOptions(
                env.getSkyframeBuildView().getBuildConfiguration().getOptions(), env.getReporter());
      }

      if (options.serverChecksumOverride != null) {
        if (mode != RemoteAnalysisCacheMode.DOWNLOAD) {
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
        this.blazeInstallMD5 = options.serverChecksumOverride;
      } else {
        this.blazeInstallMD5 = requireNonNull(env.getDirectories().getInstallMD5());
      }
      this.distinguisher = options.analysisCacheKeyDistinguisherForTesting;
      this.useFakeStampData = env.getUseFakeStampData();

      var workspaceInfoFromDiff = env.getWorkspaceInfoFromDiff();
      if (workspaceInfoFromDiff == null) {
        // If there is no workspace info, we cannot confidently version the nodes. Use the min
        // version as a sentinel.
        this.evaluatingVersion = IntVersion.of(Long.MIN_VALUE);
        this.snapshot = Optional.empty();
      } else {
        this.evaluatingVersion = workspaceInfoFromDiff.getEvaluatingVersion();
        this.snapshot = workspaceInfoFromDiff.getSnapshot();
      }
      RemoteAnalysisCachingServicesSupplier servicesSupplier =
          env.getBlazeWorkspace().remoteAnalysisCachingServicesSupplier();
      ClientId clientId =
          this.snapshot.orElse(new LongVersionClientId(this.evaluatingVersion.getVal()));
      listener.setClientId(clientId);
      servicesSupplier.configure(options, clientId, env.getCommandId().toString(), jsonLogWriter);
      this.fingerprintValueServiceFuture = servicesSupplier.getFingerprintValueService();
      this.analysisCacheClient = servicesSupplier.getAnalysisCacheClient();
      this.metadataWriter = servicesSupplier.getMetadataWriter();
    }

    HashCode getBlazeInstallMD5() {
      return blazeInstallMD5;
    }

    IntVersion getEvaluatingVersion() {
      return evaluatingVersion;
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
    public FrontierNodeVersion getSkyValueVersion() {
      if (frontierNodeVersionSingleton == null) {
        synchronized (this) {
          // Avoid re-initializing the value for subsequent threads with double checked locking.
          if (frontierNodeVersionSingleton == null) {
            frontierNodeVersionSingleton =
                new FrontierNodeVersion(
                    topLevelBuildOptions.checksum(),
                    blazeInstallMD5,
                    evaluatingVersion,
                    nullToEmpty(distinguisher),
                    useFakeStampData,
                    snapshot);
            logger.atInfo().log(
                "Remote analysis caching SkyValue version: %s (actual evaluating version: %s)",
                frontierNodeVersionSingleton, evaluatingVersion);
            listener.recordSkyValueVersion(frontierNodeVersionSingleton);
          }
        }
      }
      return frontierNodeVersionSingleton;
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

    private void setTopLevelBuildOptions(BuildOptions buildOptions) {
      this.topLevelBuildOptions = buildOptions;
    }

    @Override
    public Collection<Label> getTopLevelTargets() {
      return topLevelTargets;
    }

    @Nullable
    private AnalysisCacheInvalidator getAnalysisCacheInvalidator() throws InterruptedException {
      AnalysisCacheInvalidator localRef = analysisCacheInvalidator;
      if (localRef == null) {
        synchronized (this) {
          localRef = analysisCacheInvalidator;
          if (localRef == null) {
            ObjectCodecs codecs;
            FingerprintValueService fingerprintService;
            RemoteAnalysisCacheClient client;
            try (SilentCloseable unused =
                Profiler.instance().profile("initializeInvalidationLookupDeps")) {
              client = getAnalysisCacheClient();
              if (client == null) {
                return null;
              }
              codecs = getObjectCodecs();
              fingerprintService = getFingerprintValueService();
            }
            localRef =
                new AnalysisCacheInvalidator(
                    client,
                    codecs,
                    fingerprintService,
                    getSkyValueVersion(),
                    listener.getClientId(),
                    eventHandler);
            analysisCacheInvalidator = localRef;
          }
        }
      }
      return localRef;
    }
  }
}
