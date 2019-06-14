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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/** Handles the Merging of ParsedAndroidData. */
class AndroidDataMerger {

  public static class MergeConflictException extends UserException {

    private MergeConflictException(String message) {
      super(message);
    }

    static MergeConflictException withMessage(String message) {
      return new MergeConflictException(message);
    }
  }

  private static final Logger logger = Logger.getLogger(AndroidDataMerger.class.getCanonicalName());

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
      if (one.getPath().equals(two.getPath())) {
        return true;
      }
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
                      "Filesystem size of %s (%s) or %s (%s) is inconsistent with bytes read %s.",
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
  private final AndroidDataDeserializer deserializer;

  /** Creates a merger with no path deduplication and a default {@link ExecutorService}. */
  @VisibleForTesting
  static AndroidDataMerger createWithDefaults() {
    return createWithDefaultThreadPool(NoopSourceChecker.create());
  }

  /** Creates a merger with a custom deduplicator and a default {@link ExecutorService}. */
  @VisibleForTesting
  static AndroidDataMerger createWithDefaultThreadPool(SourceChecker deDuplicator) {
    return new AndroidDataMerger(
        deDuplicator,
        MoreExecutors.newDirectExecutorService(),
        AndroidParsedDataDeserializer.create());
  }

  /** Creates a merger with a file contents hashing deduplicator. */
  static AndroidDataMerger createWithPathDeduplictor(
      ListeningExecutorService executorService,
      AndroidDataDeserializer deserializer,
      SourceChecker checker) {
    return new AndroidDataMerger(checker, executorService, deserializer);
  }

  private AndroidDataMerger(
      SourceChecker deDuplicator,
      ListeningExecutorService executorService,
      AndroidDataDeserializer deserializer) {
    this.deDuplicator = deDuplicator;
    this.executorService = executorService;
    this.deserializer = deserializer;
  }

  /**
   * Loads a list of dependency {@link SerializedAndroidData} and merge with the primary {@link
   * ParsedAndroidData}.
   *
   * @see AndroidDataMerger#merge(ParsedAndroidData, ParsedAndroidData, UnvalidatedAndroidData,
   *     boolean, boolean) for details.
   */
  UnwrittenMergedAndroidData loadAndMerge(
      List<? extends SerializedAndroidData> transitive,
      List<? extends SerializedAndroidData> direct,
      ParsedAndroidData primary,
      Path primaryManifest,
      boolean allowPrimaryOverrideAll,
      boolean throwOnResourceConflict) {
    Stopwatch timer = Stopwatch.createStarted();
    try {
      logger.fine(
          String.format("Merged dependencies read in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      timer.reset().start();
      return doMerge(
          ParsedAndroidData.loadedFrom(transitive, executorService, deserializer),
          ParsedAndroidData.loadedFrom(direct, executorService, deserializer),
          primary,
          primaryManifest,
          allowPrimaryOverrideAll,
          throwOnResourceConflict);
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
   * @throws MergingException if there are issues with parsing resources from primaryData.
   * @throws MergeConflictException if there are merge conflicts
   */
  UnwrittenMergedAndroidData merge(
      ParsedAndroidData transitive,
      ParsedAndroidData direct,
      UnvalidatedAndroidData primaryData,
      boolean allowPrimaryOverrideAll,
      boolean throwOnResourceConflict) {
    try {
      // Extract the primary resources.
      ParsedAndroidData parsedPrimary = ParsedAndroidData.from(primaryData);
      return doMerge(
          transitive,
          direct,
          parsedPrimary,
          primaryData.getManifest(),
          allowPrimaryOverrideAll,
          throwOnResourceConflict);
    } catch (IOException e) {
      throw MergingException.wrapException(e);
    }
  }

  UnwrittenMergedAndroidData doMerge(
      ParsedAndroidData transitive,
      ParsedAndroidData direct,
      ParsedAndroidData parsedPrimary,
      Path primaryManifest,
      boolean allowPrimaryOverrideAll,
      boolean throwOnResourceConflict) {

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
    for (Map.Entry<DataKey, DataResource> entry : parsedPrimary.iterateOverwritableEntries()) {
      if (direct.containsOverwritable(entry.getKey())) {
        primaryConsumers.overwritingConsumer.accept(
            entry.getKey(), entry.getValue().overwrite(direct.getOverwritable(entry.getKey())));
      } else {
        primaryConsumers.overwritingConsumer.accept(entry.getKey(), entry.getValue());
      }
    }

    for (Map.Entry<DataKey, DataResource> entry : direct.iterateOverwritableEntries()) {
      // Direct dependencies are simply overwritten, no conflict.
      if (!parsedPrimary.containsOverwritable(entry.getKey())) {
        transitiveConsumers.overwritingConsumer.accept(entry.getKey(), entry.getValue());
      }
    }
    for (Map.Entry<DataKey, DataResource> entry : transitive.iterateOverwritableEntries()) {
      // If the primary is considered to be intentional (usually at the binary level),
      // skip.
      if (allowPrimaryOverrideAll && parsedPrimary.containsOverwritable(entry.getKey())) {
        continue;
      }
      // If a transitive value is in the direct map, report a conflict, as it is commonly
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
        transitiveConsumers.overwritingConsumer.accept(entry.getKey(), entry.getValue());
      }
    }

    // combining resources
    for (Map.Entry<DataKey, DataResource> entry : parsedPrimary.iterateCombiningEntries()) {
      primaryConsumers.combiningConsumer.accept(entry.getKey(), entry.getValue());
    }
    for (Map.Entry<DataKey, DataResource> entry : direct.iterateCombiningEntries()) {
      if (parsedPrimary.containsCombineable(entry.getKey())) {
        // If it is in the primary, add it to the primary to be combined.
        primaryConsumers.combiningConsumer.accept(entry.getKey(), entry.getValue());
      } else {
        // If the combining asset is not in the primary, put it into the transitive.
        transitiveConsumers.combiningConsumer.accept(entry.getKey(), entry.getValue());
      }
    }
    for (Map.Entry<DataKey, DataResource> entry : transitive.iterateCombiningEntries()) {
      if (parsedPrimary.containsCombineable(entry.getKey())) {
        primaryConsumers.combiningConsumer.accept(entry.getKey(), entry.getValue());
      } else {
        transitiveConsumers.combiningConsumer.accept(entry.getKey(), entry.getValue());
      }
    }

    // assets
    for (Map.Entry<DataKey, DataAsset> entry : parsedPrimary.iterateAssetEntries()) {
      if (direct.containsAsset(entry.getKey())) {
        primaryConsumers.assetConsumer.accept(
            entry.getKey(), entry.getValue().overwrite(direct.getAsset(entry.getKey())));
      } else {
        primaryConsumers.assetConsumer.accept(entry.getKey(), entry.getValue());
      }
    }

    for (Map.Entry<DataKey, DataAsset> entry : direct.iterateAssetEntries()) {
      // Direct dependencies are simply overwritten, no conflict.
      if (!parsedPrimary.containsAsset(entry.getKey())) {
        transitiveConsumers.assetConsumer.accept(entry.getKey(), entry.getValue());
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
        transitiveConsumers.assetConsumer.accept(entry.getKey(), entry.getValue());
      }
    }
    final UnwrittenMergedAndroidData unwrittenMergedAndroidData = UnwrittenMergedAndroidData.of(
        primaryManifest, primaryBuilder.build(), transitiveBuilder.build(), conflicts);

    try {
      List<String> messages = unwrittenMergedAndroidData
          .asConflictMessagesIfValidWith(deDuplicator);

      if (!messages.isEmpty()) {
        if (throwOnResourceConflict) {
          throw MergeConflictException.withMessage(Joiner.on("\n").join(messages));
        } else {
          logger.warning(Joiner.on("\n").join(messages));
        }
      }
    } catch (IOException e) {
      throw MergingException.wrapException(e);
    }

    return unwrittenMergedAndroidData;
  }
}
