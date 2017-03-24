// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.android.annotations.Nullable;
import com.android.builder.core.VariantType;
import com.android.ide.common.internal.PngCruncher;
import com.android.ide.common.res2.MergingException;
import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/** Collects all the functionality for an action to merge resources. */
// TODO(bazel-team): Turn into an instance object, in order to use an external ExecutorService.
public class AndroidResourceMerger {
  static final Logger logger = Logger.getLogger(AndroidResourceProcessor.class.getName());

  /** Merges all secondary resources with the primary resources. */
  public static MergedAndroidData mergeData(
      final ParsedAndroidData primary,
      final Path primaryManifest,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut,
      @Nullable AndroidResourceClassWriter rclassWriter,
      AndroidDataDeserializer deserializer)
      throws MergingException {
    Stopwatch timer = Stopwatch.createStarted();
    final ListeningExecutorService executorService =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(15));
    try (Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {
      AndroidDataMerger merger =
          AndroidDataMerger.createWithPathDeduplictor(executorService, deserializer);
      UnwrittenMergedAndroidData merged =
          mergeData(
              executorService,
              transitive,
              direct,
              primary,
              primaryManifest,
              type != VariantType.LIBRARY,
              deserializer);
      timer.reset().start();
      if (symbolsOut != null) {
        AndroidDataSerializer serializer = AndroidDataSerializer.create();
        merged.serializeTo(serializer);
        serializer.flushTo(symbolsOut);
        logger.fine(
            String.format(
                "serialize merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        timer.reset().start();
      }
      if (rclassWriter != null) {
        merged.writeResourceClass(rclassWriter);
        logger.fine(
            String.format("write classes finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        timer.reset().start();
      }
      AndroidDataWriter writer =
          AndroidDataWriter.createWith(
              resourcesOut.getParent(), resourcesOut, assetsOut, cruncher, executorService);
      return merged.write(writer);
    } catch (IOException e) {
      throw MergingException.wrapException(e).build();
    } finally {
      logger.fine(
          String.format("write merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }

  public static UnwrittenMergedAndroidData mergeData(
      ListeningExecutorService executorService,
      List<? extends SerializedAndroidData> transitive,
      List<? extends SerializedAndroidData> direct,
      ParsedAndroidData primary,
      Path primaryManifest,
      boolean allowPrimaryOverrideAll,
      AndroidDataDeserializer deserializer)
      throws MergingException {
    Stopwatch timer = Stopwatch.createStarted();
    try {
      AndroidDataMerger merger =
          AndroidDataMerger.createWithPathDeduplictor(executorService, deserializer);
      return merger.loadAndMerge(
          transitive, direct, primary, primaryManifest, allowPrimaryOverrideAll);
    } finally {
      logger.fine(String.format("merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }

  /**
   * Merges all secondary resources with the primary resources, given that the primary resources
   * have not yet been parsed and serialized.
   */
  public static MergedAndroidData mergeData(
      final UnvalidatedAndroidData primary,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut,
      final List<String> filteredResources)
      throws MergingException {
    try {
      final ParsedAndroidData parsedPrimary = ParsedAndroidData.from(primary);
      return mergeData(
          parsedPrimary,
          primary.getManifest(),
          direct,
          transitive,
          resourcesOut,
          assetsOut,
          cruncher,
          type,
          symbolsOut,
          null /* rclassWriter */,
          AndroidDataDeserializer.withFilteredResources(filteredResources));
    } catch (IOException e) {
      throw MergingException.wrapException(e).build();
    }
  }

  /**
   * Merges all secondary resources with the primary resources, given that the primary resources
   * have been separately parsed and serialized.
   */
  public static MergedAndroidData mergeData(
      final SerializedAndroidData primary,
      final Path primaryManifest,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut,
      @Nullable final AndroidResourceClassWriter rclassWriter)
      throws MergingException {
    final ParsedAndroidData.Builder primaryBuilder = ParsedAndroidData.Builder.newBuilder();
    final AndroidDataDeserializer deserializer = AndroidDataDeserializer.create();
    primary.deserialize(deserializer, primaryBuilder.consumers());
    ParsedAndroidData primaryData = primaryBuilder.build();
    return mergeData(
        primaryData,
        primaryManifest,
        direct,
        transitive,
        resourcesOut,
        assetsOut,
        cruncher,
        type,
        symbolsOut,
        rclassWriter,
        deserializer);
  }
}


