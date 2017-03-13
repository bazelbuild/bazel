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
package com.google.devtools.build.android;

import com.android.ide.common.res2.MergingException;
import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.ParsedAndroidData.Builder;
import com.google.devtools.build.android.ParsedAndroidData.ParsedAndroidDataBuildingPathWalker;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/** Handles the Merging of ParsedAndroidData. */
class AndroidDataMerger {

  private static final Logger logger = Logger.getLogger(AndroidDataMerger.class.getCanonicalName());

  private final class ParseDependencyDataTask implements Callable<Boolean> {

    private final SerializedAndroidData dependency;

    private final Builder targetBuilder;

    private ParseDependencyDataTask(SerializedAndroidData dependency, Builder targetBuilder) {
      this.dependency = dependency;
      this.targetBuilder = targetBuilder;
    }

    @Override
    public Boolean call() throws Exception {
      final Builder parsedDataBuilder = ParsedAndroidData.Builder.newBuilder();
      try {
        dependency.deserialize(deserializer, parsedDataBuilder.consumers());
      } catch (DeserializationException e) {
        if (!e.isLegacy()) {
          throw MergingException.wrapException(e).build();
        }
        logger.fine(
            String.format(
                "\u001B[31mDEPRECATION:\u001B[0m Legacy resources used for %s",
                dependency.getLabel()));
        // Legacy android resources -- treat them as direct dependencies.
        dependency.walk(ParsedAndroidDataBuildingPathWalker.create(parsedDataBuilder));
      }
      // The builder isn't threadsafe, so synchronize the copyTo call.
      synchronized (targetBuilder) {
        // All the resources are sorted before writing, so they can be aggregated in
        // whatever order here.
        parsedDataBuilder.copyTo(targetBuilder);
      }
      // Had to return something?
      return Boolean.TRUE;
    }
  }

  /** Interface for comparing paths. */
  interface SourceChecker {
    boolean checkEquality(DataSource one, DataSource two) throws IOException;
  }

  /** Compares two paths by the contents of the files. */
  static class ContentComparingChecker implements SourceChecker {

    static SourceChecker create() {
      return new ContentComparingChecker();
    }

    @Override
    public boolean checkEquality(DataSource one, DataSource two) throws IOException {
      // TODO(corysmith): Is there a filesystem hash we can use?
      if (one.getFileSize() != two.getFileSize()) {
        return false;
      }
      try (final InputStream oneStream = one.newBufferedInputStream();
          final InputStream twoStream = two.newBufferedInputStream()) {
        int bytesRead = 0;
        while (true) {
          int oneByte = oneStream.read();
          int twoByte = twoStream.read();
          bytesRead++;
          if (oneByte == -1 || twoByte == -1) {
            if (oneByte == twoByte) {
              return true;
            } else {
              // getFileSize did not return correct size.
              logger.severe(
                  String.format(
                      "Filesystem size of %s (%s) or %s (%s) is inconsistant with bytes read %s.",
                      one, one.getFileSize(), two, two.getFileSize(), bytesRead));
              return false;
            }
          }
          if (oneByte != twoByte) {
            return false;
          }
        }
      }
    }
  }

  static class NoopSourceChecker implements SourceChecker {
    static SourceChecker create() {
      return new NoopSourceChecker();
    }

    @Override
    public boolean checkEquality(DataSource one, DataSource two) {
      return false;
    }
  }

  private final SourceChecker deDuplicator;
  private final ListeningExecutorService executorService;
  private final AndroidDataDeserializer deserializer = AndroidDataDeserializer.create();

  /** Creates a merger with no path deduplication and a default {@link ExecutorService}. */
  static AndroidDataMerger createWithDefaults() {
    return createWithDefaultThreadPool(NoopSourceChecker.create());
  }

  /** Creates a merger with a custom deduplicator and a default {@link ExecutorService}. */
  static AndroidDataMerger createWithDefaultThreadPool(SourceChecker deDuplicator) {
    return new AndroidDataMerger(deDuplicator, MoreExecutors.newDirectExecutorService());
  }

  /** Creates a merger with a custom deduplicator and an {@link ExecutorService}. */
  static AndroidDataMerger create(
      SourceChecker deDuplicator, ListeningExecutorService executorService) {
    return new AndroidDataMerger(deDuplicator, executorService);
  }

  /** Creates a merger with a file contents hashing deduplicator. */
  static AndroidDataMerger createWithPathDeduplictor(
      ListeningExecutorService executorService) {
    return create(ContentComparingChecker.create(), executorService);
  }

  private AndroidDataMerger(SourceChecker deDuplicator, ListeningExecutorService executorService) {
    this.deDuplicator = deDuplicator;
    this.executorService = executorService;
  }

  /**
   * Loads a list of dependency {@link SerializedAndroidData} and merge with the primary {@link
   * ParsedAndroidData}.
   *
   * @see AndroidDataMerger#merge(ParsedAndroidData, ParsedAndroidData, UnvalidatedAndroidData,
   *     boolean) for details.
   */
  UnwrittenMergedAndroidData loadAndMerge(
      List<? extends SerializedAndroidData> transitive,
      List<? extends SerializedAndroidData> direct,
      ParsedAndroidData primary,
      Path primaryManifest,
      boolean allowPrimaryOverrideAll)
      throws MergingException {
    Stopwatch timer = Stopwatch.createStarted();
    try {
      final ParsedAndroidData.Builder directBuilder = ParsedAndroidData.Builder.newBuilder();
      final ParsedAndroidData.Builder transitiveBuilder = ParsedAndroidData.Builder.newBuilder();
      final List<ListenableFuture<Boolean>> tasks = new ArrayList<>();
      for (final SerializedAndroidData dependency : direct) {
        tasks.add(executorService.submit(new ParseDependencyDataTask(dependency, directBuilder)));
      }
      for (final SerializedAndroidData dependency : transitive) {
        tasks.add(
            executorService.submit(new ParseDependencyDataTask(dependency, transitiveBuilder)));
      }
      // Wait for all the parsing to complete.
      FailedFutureAggregator<MergingException> aggregator =
          FailedFutureAggregator.createForMergingExceptionWithMessage(
              "Failure(s) during dependency parsing");
      aggregator.aggregateAndMaybeThrow(tasks);
      logger.fine(
          String.format("Merged dependencies read in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      timer.reset().start();
      return doMerge(
          transitiveBuilder.build(),
          directBuilder.build(),
          primary,
          primaryManifest,
          allowPrimaryOverrideAll);
    } finally {
      logger.fine(String.format("Resources merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }

  /**
   * Merges DataResources into an UnwrittenMergedAndroidData.
   *
   * <p>This method has two basic states, library and binary. These are distinguished by
   * allowPrimaryOverrideAll, which allows the primary data to overwrite any value in the closure, a
   * trait associated with binaries, as a binary is a leaf node. The other semantics are slightly
   * more complicated: a given resource can be overwritten only if it resides in the direct
   * dependencies of primary data. This forces an explicit simple priority for each resource,
   * instead of the more subtle semantics of multiple layers of libraries with potential overwrites.
   *
   * <p>The UnwrittenMergedAndroidData contains only one of each DataKey in both the direct and
   * transitive closure.
   *
   * <p>The merge semantics for overwriting resources (non id and styleable) are as follows:
   *
   * <pre>
   *   Key:
   *     A(): package A
   *     A(foo): package A with resource symbol foo
   *     A() -> B(): a dependency relationship of B.deps = [:A]
   *     A(),B() -> C(): a dependency relationship of C.deps = [:A,:B]
   *
   *   For android library (allowPrimaryOverrideAll = False)
   *
   *     A() -> B(foo) -> C(foo) == Valid
   *     A() -> B() -> C(foo) == Valid
   *     A() -> B() -> C(foo),D(foo) == Conflict
   *     A(foo) -> B(foo) -> C() == Conflict
   *     A(foo) -> B() -> C(foo) == Conflict
   *     A(foo),B(foo) -> C() -> D() == Conflict
   *     A() -> B(foo),C(foo) -> D() == Conflict
   *     A(foo),B(foo) -> C() -> D(foo) == Conflict
   *     A() -> B(foo),C(foo) -> D(foo) == Conflict
   *
   *   For android binary (allowPrimaryOverrideAll = True)
   *
   *     A() -> B(foo) -> C(foo) == Valid
   *     A() -> B() -> C(foo) == Valid
   *     A() -> B() -> C(foo),D(foo) == Conflict
   *     A(foo) -> B(foo) -> C() == Conflict
   *     A(foo) -> B() -> C(foo) == Valid
   *     A(foo),B(foo) -> C() -> D() == Conflict
   *     A() -> B(foo),C(foo) -> D() == Conflict
   *     A(foo),B(foo) -> C() -> D(foo) == Valid
   *     A() -> B(foo),C(foo) -> D(foo) == Valid
   * </pre>
   *
   * <p>Combining resources are much simpler -- since a combining (id and styleable) resource does
   * not get replaced when redefined, they are simply combined:
   *
   * <pre>
   *     A(foo) -> B(foo) -> C(foo) == Valid
   *
   * </pre>
   *
   * @param transitive The transitive dependencies to merge.
   * @param direct The direct dependencies to merge.
   * @param primaryData The primary data to merge against.
   * @param allowPrimaryOverrideAll Boolean that indicates if the primary data will be considered
   *     the ultimate source of truth, provided it doesn't conflict with itself.
   * @return An UnwrittenMergedAndroidData, containing DataResource objects that can be written to
   *     disk for aapt processing or serialized for future merge passes.
   * @throws MergingException if there are merge conflicts or issues with parsing resources from
   *     primaryData.
   */
  UnwrittenMergedAndroidData merge(
      ParsedAndroidData transitive,
      ParsedAndroidData direct,
      UnvalidatedAndroidData primaryData,
      boolean allowPrimaryOverrideAll)
      throws MergingException {
    try {
      // Extract the primary resources.
      ParsedAndroidData parsedPrimary = ParsedAndroidData.from(primaryData);
      return doMerge(
          transitive, direct, parsedPrimary, primaryData.getManifest(), allowPrimaryOverrideAll);
    } catch (IOException e) {
      throw MergingException.wrapException(e).build();
    }
  }

  private UnwrittenMergedAndroidData doMerge(
      ParsedAndroidData transitive,
      ParsedAndroidData direct,
      ParsedAndroidData parsedPrimary,
      Path primaryManifest,
      boolean allowPrimaryOverrideAll)
      throws MergingException {
    try {
      // Create the builders for the final parsed data.
      final ParsedAndroidData.Builder primaryBuilder = ParsedAndroidData.Builder.newBuilder();
      final ParsedAndroidData.Builder transitiveBuilder = ParsedAndroidData.Builder.newBuilder();
      final KeyValueConsumers transitiveConsumers = transitiveBuilder.consumers();
      final KeyValueConsumers primaryConsumers = primaryBuilder.consumers();

      final Set<MergeConflict> conflicts = new HashSet<>();

      // Find all internal conflicts.
      conflicts.addAll(parsedPrimary.conflicts());
      for (MergeConflict conflict : Iterables.concat(direct.conflicts(), transitive.conflicts())) {
        if (allowPrimaryOverrideAll
            && (parsedPrimary.containsOverwritable(conflict.dataKey())
                || parsedPrimary.containsAsset(conflict.dataKey()))) {
          continue;
        }
        conflicts.add(conflict);
      }

      // overwriting resources
      for (Entry<DataKey, DataResource> entry : parsedPrimary.iterateOverwritableEntries()) {
        if (direct.containsOverwritable(entry.getKey())) {
          primaryConsumers.overwritingConsumer.consume(
              entry.getKey(), entry.getValue().overwrite(direct.getOverwritable(entry.getKey())));
        } else {
          primaryConsumers.overwritingConsumer.consume(entry.getKey(), entry.getValue());
        }
      }

      for (Map.Entry<DataKey, DataResource> entry : direct.iterateOverwritableEntries()) {
        // Direct dependencies are simply overwritten, no conflict.
        if (!parsedPrimary.containsOverwritable(entry.getKey())) {
          transitiveConsumers.overwritingConsumer.consume(entry.getKey(), entry.getValue());
        }
      }
      for (Map.Entry<DataKey, DataResource> entry : transitive.iterateOverwritableEntries()) {
        // If the primary is considered to be intentional (usually at the binary level),
        // skip.
        if (allowPrimaryOverrideAll && parsedPrimary.containsOverwritable(entry.getKey())) {
          continue;
        }
        // If a transitive value is in the direct map report a conflict, as it is commonly
        // unintentional.
        if (direct.containsOverwritable(entry.getKey())) {
          conflicts.add(direct.foundResourceConflict(entry.getKey(), entry.getValue()));
        } else if (parsedPrimary.containsOverwritable(entry.getKey())) {
          // If overwriting a transitive value with a primary map, assume it's an unintentional
          // override, unless allowPrimaryOverrideAll is set. At which point, this code path
          // should not be reached.
          conflicts.add(parsedPrimary.foundResourceConflict(entry.getKey(), entry.getValue()));
        } else {
          // If it's in none of the of sources, add it.
          transitiveConsumers.overwritingConsumer.consume(entry.getKey(), entry.getValue());
        }
      }

      // combining resources
      for (Entry<DataKey, DataResource> entry : parsedPrimary.iterateCombiningEntries()) {
        primaryConsumers.combiningConsumer.consume(entry.getKey(), entry.getValue());
      }
      for (Map.Entry<DataKey, DataResource> entry : direct.iterateCombiningEntries()) {
        if (parsedPrimary.containsCombineable(entry.getKey())) {
          // If it is in the primary, add it to the primary to be combined.
          primaryConsumers.combiningConsumer.consume(entry.getKey(), entry.getValue());
        } else {
          // If the combining asset is not in the primary, put it into the transitive.
          transitiveConsumers.combiningConsumer.consume(entry.getKey(), entry.getValue());
        }
      }
      for (Map.Entry<DataKey, DataResource> entry : transitive.iterateCombiningEntries()) {
        if (parsedPrimary.containsCombineable(entry.getKey())) {
          primaryConsumers.combiningConsumer.consume(entry.getKey(), entry.getValue());
        } else {
          transitiveConsumers.combiningConsumer.consume(entry.getKey(), entry.getValue());
        }
      }

      // assets
      for (Entry<DataKey, DataAsset> entry : parsedPrimary.iterateAssetEntries()) {
        if (direct.containsAsset(entry.getKey())) {
          primaryConsumers.assetConsumer.consume(
              entry.getKey(), entry.getValue().overwrite(direct.getAsset(entry.getKey())));
        } else {
          primaryConsumers.assetConsumer.consume(entry.getKey(), entry.getValue());
        }
      }

      for (Map.Entry<DataKey, DataAsset> entry : direct.iterateAssetEntries()) {
        // Direct dependencies are simply overwritten, no conflict.
        if (!parsedPrimary.containsAsset(entry.getKey())) {
          transitiveConsumers.assetConsumer.consume(entry.getKey(), entry.getValue());
        }
      }
      for (Map.Entry<DataKey, DataAsset> entry : transitive.iterateAssetEntries()) {
        // If the primary is considered to be intentional (usually at the binary level),
        // skip.
        if (allowPrimaryOverrideAll && parsedPrimary.containsAsset(entry.getKey())) {
          continue;
        }
        // If a transitive value is in the direct map report a conflict, as it is commonly
        // unintentional.
        if (direct.containsAsset(entry.getKey())) {
          conflicts.add(direct.foundAssetConflict(entry.getKey(), entry.getValue()));
        } else if (parsedPrimary.containsAsset(entry.getKey())) {
          // If overwriting a transitive value with a primary map, assume it's an unintentional
          // override, unless allowPrimaryOverrideAll is set. At which point, this code path
          // should not be reached.
          conflicts.add(parsedPrimary.foundAssetConflict(entry.getKey(), entry.getValue()));
        } else {
          // If it's in none of the of sources, add it.
          transitiveConsumers.assetConsumer.consume(entry.getKey(), entry.getValue());
        }
      }

      if (!conflicts.isEmpty()) {
        List<String> messages = new ArrayList<>();
        for (MergeConflict conflict : conflicts) {
          if (conflict.isValidWith(deDuplicator)) {
            messages.add(conflict.toConflictMessage());
          }
        }
        if (!messages.isEmpty()) {
          // TODO(corysmith): Turn these into errors.
          logger.warning(Joiner.on("").join(messages));
        }
      }
      return UnwrittenMergedAndroidData.of(
          primaryManifest, primaryBuilder.build(), transitiveBuilder.build());
    } catch (IOException e) {
      throw MergingException.wrapException(e).build();
    }
  }
}
