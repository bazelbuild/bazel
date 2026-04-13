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

import com.google.common.base.Ascii;
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
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
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
import com.google.devtools.build.lib.skyframe.WorkspaceInfoFromDiff;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SkycacheMetadataParams;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.LongVersionClientId;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheManager.AnalysisDeps;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions.RemoteAnalysisCacheMode;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Root.RootCodecDependencies;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** Factory for {@link RemoteAnalysisCacheManager}. */
public final class RemoteAnalysisCacheFactory {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private RemoteAnalysisCacheFactory() {}

  public static AnalysisDeps create(
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
        || options.getMode() == RemoteAnalysisCacheMode.OFF) {
      RemoteAnalysisCacheDeps disabledDeps = RemoteAnalysisCacheDeps.createDisabled();
      return new AnalysisDeps(
          RemoteAnalysisCacheManager.createDisabled(), disabledDeps, disabledDeps);
    }

    // Set up active directory matcher

    Optional<PathFragmentPrefixTrie> maybeActiveDirectoriesMatcherFromFlags =
        finalizeActiveDirectoriesMatcher(env, maybeActiveDirectoriesMatcher, options.getMode());
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

    byte[] starlarkSemanticsFingerprint =
        BuildLanguageOptions.stableFingerprint(
                env.getSkyframeExecutor()
                    .getEffectiveStarlarkSemantics(
                        env.getOptions().getOptions(BuildLanguageOptions.class)))
            .toByteArray();

    FrontierNodeVersion frontierNodeVersion =
        new FrontierNodeVersion(
            topLevelOptions.checksum(),
            blazeInstallMD5,
            starlarkSemanticsFingerprint,
            workspaceInfoFromDiff.getEvaluatingVersion(),
            nullToEmpty(options.getAnalysisCacheKeyDistinguisherForTesting()),
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
        skycacheMetadataParams != null && options.getAnalysisCacheEnableMetadataQueries();

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
            options.getMode(),
            options.getAnalysisCacheBailOnMissingFingerprint(),
            options.getSkycacheMinimizeMemory(),
            servicesSupplier,
            env.getRemoteAnalysisCachingEventListener(),
            jsonLogWriter,
            objectCodecs,
            frontierNodeVersion,
            activeDirectoriesMatcher,
            options.getSerializedFrontierProfile());

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
            options.getMode(),
            areMetadataQueriesEnabled,
            env.getReporter(),
            skycacheMetadataParams,
            servicesSupplier.getAnalysisCacheClient(),
            analysisCacheInvalidator,
            topLevelTargets,
            activeDirectoriesMatcher,
            options.getSkycacheMinimizeMemory());

    // Bail out if needed

    return switch (options.getMode()) {
      case RemoteAnalysisCacheMode.DUMP_UPLOAD_MANIFEST_ONLY, RemoteAnalysisCacheMode.UPLOAD ->
          new AnalysisDeps(manager, deps, deps);
      case RemoteAnalysisCacheMode.DOWNLOAD -> {
        RemoteAnalysisCacheClient analysisCacheClient;
        try (SilentCloseable unused = Profiler.instance().profile("initAnalysisCacheClient")) {
          analysisCacheClient = deps.getAnalysisCacheClient();
        }
        if (analysisCacheClient == null) {
          if (Strings.isNullOrEmpty(options.getAnalysisCacheService())) {
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
                            + " local evaluation."));
          }
          yield new AnalysisDeps(
              RemoteAnalysisCacheManager.createDisabled(),
              RemoteAnalysisCacheDeps.createDisabled(),
              RemoteAnalysisCacheDeps.createDisabled());
        }
        yield new AnalysisDeps(manager, deps, deps);
      }
      default ->
          throw new IllegalStateException("Unknown RemoteAnalysisCacheMode: " + options.getMode());
    };
  }

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
    if (options.getServerChecksumOverride() == null) {
      return requireNonNull(env.getDirectories().getInstallMD5());
    }

    if (options.getMode() != RemoteAnalysisCacheMode.DOWNLOAD) {
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
                    options.getServerChecksumOverride(), env.getDirectories().getInstallMD5())));

    return options.getServerChecksumOverride();
  }

  @Nullable
  private static RemoteAnalysisJsonLogWriter createJsonLogWriterMaybe(
      CommandEnvironment env, RemoteAnalysisCachingOptions options) throws AbruptExitException {
    if (options.getJsonLog() == null) {
      return null;
    }
    try {
      InstrumentationOutput jsonLog =
          env.getRuntime()
              .getInstrumentationOutputFactory()
              .createInstrumentationOutput(
                  "remote_cache_jsonlog",
                  PathFragment.create(options.getJsonLog()),
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
          .handle(
              Event.info(String.format("Writing Skycache JSON log to '%s'", options.getJsonLog())));
      return result;
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "Cannot open remote analysis JSON log file '%s': %s",
                          options.getJsonLog(), e.getMessage()))
                  .setRemoteAnalysisCaching(
                      RemoteAnalysisCaching.newBuilder()
                          .setCode(RemoteAnalysisCaching.Code.CANNOT_OPEN_LOG_FILE))
                  .build()));
    }
  }

  private static ImmutableSet<String> getConfigurationOptionsAsStrings(BuildOptions targetOptions) {
    ImmutableSet.Builder<String> allOptionsAsStringsBuilder = new ImmutableSet.Builder<>();

    // Collect a list of BuildOptions, excluding TestOptions.
    targetOptions.getStarlarkOptions().keySet().stream()
        .map(Object::toString)
        .forEach(allOptionsAsStringsBuilder::add);
    for (FragmentOptions fragmentOptions : targetOptions.getNativeOptions()) {
      if (fragmentOptions instanceof TestConfiguration.TestOptions) {
        continue;
      }
      fragmentOptions.asMap().keySet().forEach(allOptionsAsStringsBuilder::add);
    }
    return allOptionsAsStringsBuilder.build();
  }
}
