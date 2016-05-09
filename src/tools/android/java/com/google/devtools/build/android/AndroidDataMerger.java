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

import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.ParsedAndroidData.ParsedAndroidDataBuildingPathWalker;

import com.android.ide.common.res2.MergingException;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributeView;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Handles the Merging of ParsedAndroidData.
 */
public class AndroidDataMerger {

  private static final Logger logger = Logger.getLogger(AndroidDataMerger.class.getCanonicalName());

  /** Interface for comparing paths. */
  interface SourceChecker {
    boolean checkEquality(Path one, Path two) throws IOException;
  }

  /** Compares two paths by the contents of the files. */
  static class ContentComparingChecker implements SourceChecker {

    static SourceChecker create() {
      return new ContentComparingChecker();
    }

    @Override
    public boolean checkEquality(Path one, Path two) throws IOException {
      // TODO(corysmith): Is there a filesystem hash we can use?
      if (getFileSize(one) != getFileSize(two)) {
        return false;
      }
      try (final InputStream oneStream = new BufferedInputStream(Files.newInputStream(one));
          final InputStream twoStream = new BufferedInputStream(Files.newInputStream(two))) {
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
                      one,
                      getFileSize(one),
                      two,
                      getFileSize(two),
                      bytesRead));
              return false;
            }
          }
          if (oneByte != twoByte) {
            return false;
          }
        }
      }
    }

    private long getFileSize(Path path) throws IOException {
      return Files.getFileAttributeView(path, BasicFileAttributeView.class).readAttributes().size();
    }
  }

  static class NoopSourceChecker implements SourceChecker {
    static SourceChecker create() {
      return new NoopSourceChecker();
    }

    @Override
    public boolean checkEquality(Path one, Path two) {
      return false;
    }
  }

  private final SourceChecker deDuplicator;

  /** Creates a merger with no path deduplication. */
  public static AndroidDataMerger create() {
    return new AndroidDataMerger(NoopSourceChecker.create());
  }

  /** Creates a merger with a custom deduplicator. */
  public static AndroidDataMerger create(SourceChecker deDuplicator) {
    return new AndroidDataMerger(deDuplicator);
  }

  /** Creates a merger with a file contents hashing deduplicator. */
  public static AndroidDataMerger createWithPathDeduplictor() {
    return create(ContentComparingChecker.create());
  }

  private AndroidDataMerger(SourceChecker deDuplicator) {
    this.deDuplicator = deDuplicator;
  }

  /**
   * Merges a list of {@link DependencyAndroidData} with a {@link UnvalidatedAndroidData}.
   *
   * @see AndroidDataMerger#merge(ParsedAndroidData, ParsedAndroidData, UnvalidatedAndroidData,
   * boolean) for details.
   */
  UnwrittenMergedAndroidData merge(
      List<DependencyAndroidData> transitive,
      List<DependencyAndroidData> direct,
      UnvalidatedAndroidData primary,
      boolean allowPrimaryOverrideAll)
      throws IOException, MergingException {
    Stopwatch timer = Stopwatch.createStarted();
    try {
      final ParsedAndroidData.Builder directBuilder = ParsedAndroidData.Builder.newBuilder();
      final ParsedAndroidData.Builder transitiveBuilder = ParsedAndroidData.Builder.newBuilder();
      final AndroidDataSerializer serializer = AndroidDataSerializer.create();
      for (final DependencyAndroidData dependency : direct) {
        parseDependencyData(directBuilder, serializer, dependency);
      }
      for (final DependencyAndroidData dependency : transitive) {
        parseDependencyData(transitiveBuilder, serializer, dependency);
      }
      logger.fine(
          String.format("Merged dependencies read in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      timer.reset().start();
      return merge(
          transitiveBuilder.build(), directBuilder.build(), primary, allowPrimaryOverrideAll);
    } finally {
      logger.fine(String.format("Resources merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }

  private void parseDependencyData(
      final ParsedAndroidData.Builder parsedDataBuilder,
      final AndroidDataSerializer serializer,
      final DependencyAndroidData dependency)
      throws IOException, MergingException {
    try {
      dependency.deserialize(serializer, parsedDataBuilder.consumers());
    } catch (DeserializationException e) {
      if (!e.isLegacy()) {
        throw new MergingException(e);
      }
      //TODO(corysmith): List the offending a target here.
      logger.warning(
          String.format(
              "\u001B[31mDEPRECATION:\u001B[0m Legacy resources used for %s",
              dependency.getManifest()));
      // Legacy android resources -- treat them as direct dependencies.
      dependency.walk(ParsedAndroidDataBuildingPathWalker.create(parsedDataBuilder));
    }
  }

  /**
   * Merges DataResources into an UnwrittenMergedAndroidData.
   * <p>
   * This method has two basic states, library and binary. These are distinguished by
   * allowPrimaryOverrideAll, which allows the primary data to overwrite any value in the closure, a
   * trait associated with binaries, as a binary is a leaf node. The other semantics are slightly
   * more complicated: a given resource can be overwritten only if it resides in the direct
   * dependencies of primary data. This forces an explicit simple priority for each resource,
   * instead of the more subtle semantics of multiple layers of libraries with potential overwrites.
   * <p>
   * The UnwrittenMergedAndroidData contains only one of each DataKey in both the direct and
   * transitive closure.
   *
   * The merge semantics are as follows: <pre>
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
   * @param transitive The transitive dependencies to merge.
   * @param direct The direct dependencies to merge.
   * @param primaryData The primary data to merge against.
   * @param allowPrimaryOverrideAll Boolean that indicates if the primary data will be considered
   *    the ultimate source of truth, provided it doesn't conflict with itself.
   * @return An UnwrittenMergedAndroidData, containing DataResource objects that can be written to
   *    disk for aapt processing or serialized for future merge passes.
   * @throws MergingException if there are merge conflicts or issues with parsing resources from
   *    Primary.
   */
  UnwrittenMergedAndroidData merge(
      ParsedAndroidData transitive,
      ParsedAndroidData direct,
      UnvalidatedAndroidData primaryData,
      boolean allowPrimaryOverrideAll)
      throws MergingException {

    try {
      // Extract the primary resources.
      ParsedAndroidData primary = ParsedAndroidData.from(primaryData);

      Map<DataKey, DataResource> overwritableDeps = new HashMap<>();
      Map<DataKey, DataAsset> assets = new HashMap<>();

      Set<MergeConflict> conflicts = new HashSet<>();
      conflicts.addAll(primary.conflicts());
      for (MergeConflict conflict : Iterables.concat(direct.conflicts(), transitive.conflicts())) {
        if (allowPrimaryOverrideAll
            && (primary.containsOverwritable(conflict.dataKey())
                || primary.containsAsset(conflict.dataKey()))) {
          continue;
        }
        conflicts.add(conflict);
      }

      // resources
      for (Map.Entry<DataKey, DataResource> entry : direct.iterateOverwritableEntries()) {
        // Direct dependencies are simply overwritten, no conflict.
        if (!primary.containsOverwritable(entry.getKey())) {
          overwritableDeps.put(entry.getKey(), entry.getValue());
        }
      }
      for (Map.Entry<DataKey, DataResource> entry : transitive.iterateOverwritableEntries()) {
        // If the primary is considered to be intentional (usually at the binary level),
        // skip.
        if (primary.containsOverwritable(entry.getKey()) && allowPrimaryOverrideAll) {
          continue;
        }
        // If a transitive value is in the direct map report a conflict, as it is commonly
        // unintentional.
        if (direct.containsOverwritable(entry.getKey())) {
          conflicts.add(direct.foundResourceConflict(entry.getKey(), entry.getValue()));
        } else if (primary.containsOverwritable(entry.getKey())) {
          // If overwriting a transitive value with a primary map, assume it's an unintentional
          // override, unless allowPrimaryOverrideAll is set. At which point, this code path
          // should not be reached.
          conflicts.add(primary.foundResourceConflict(entry.getKey(), entry.getValue()));
        } else {
          // If it's in none of the of sources, add it.
          overwritableDeps.put(entry.getKey(), entry.getValue());
        }
      }

      // assets
      for (Map.Entry<DataKey, DataAsset> entry : direct.iterateAssetEntries()) {
        // Direct dependencies are simply overwritten, no conflict.
        if (!primary.containsAsset(entry.getKey())) {
          assets.put(entry.getKey(), entry.getValue());
        }
      }
      for (Map.Entry<DataKey, DataAsset> entry : transitive.iterateAssetEntries()) {
        // If the primary is considered to be intentional (usually at the binary level),
        // skip.
        if (primary.containsAsset(entry.getKey()) && allowPrimaryOverrideAll) {
          continue;
        }
        // If a transitive value is in the direct map report a conflict, as it is commonly
        // unintentional.
        if (direct.containsAsset(entry.getKey())) {
          conflicts.add(direct.foundAssetConflict(entry.getKey(), entry.getValue()));
        } else if (primary.containsAsset(entry.getKey())) {
          // If overwriting a transitive value with a primary map, assume it's an unintentional
          // override, unless allowPrimaryOverrideAll is set. At which point, this code path
          // should not be reached.
          conflicts.add(primary.foundAssetConflict(entry.getKey(), entry.getValue()));
        } else {
          // If it's in none of the of sources, add it.
          assets.put(entry.getKey(), entry.getValue());
        }
      }

      if (!conflicts.isEmpty()) {
        List<String> messages = new ArrayList<>();
        for (MergeConflict conflict : conflicts) {
          if (!conflict.first().equals(conflict.second())
              && !deDuplicator.checkEquality(
                  conflict.first().source(), conflict.second().source())) {
            messages.add(conflict.toConflictMessage());
          }
        }
        if (!messages.isEmpty()) {
          // TODO(corysmith): Turn these into errors.
          logger.warning(Joiner.on("").join(messages));
        }
      }

      return UnwrittenMergedAndroidData.of(
          primaryData.getManifest(),
          primary,
          ParsedAndroidData.of(
              ImmutableSet.<MergeConflict>of(),
              ImmutableMap.copyOf(overwritableDeps),
              direct.mergeCombining(transitive),
              ImmutableMap.copyOf(assets)));
    } catch (IOException e) {
      throw new MergingException(e);
    }
  }
}
