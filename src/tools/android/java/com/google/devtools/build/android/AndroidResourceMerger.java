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
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.AndroidDataMerger.ContentComparingChecker;
import com.google.devtools.build.android.AndroidDataMerger.SourceChecker;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/** Collects all the functionality for an action to merge resources. */
// TODO(bazel-team): Turn into an instance object, in order to use an external ExecutorService.
public class AndroidResourceMerger {

  /** Performs a merge of compiled android data. */
  static Path mergeDataToSymbols(
      ParsedAndroidData primary,
      Path manifest,
      ImmutableList<SerializedAndroidData> direct,
      ImmutableList<SerializedAndroidData> transitive,
      VariantType packageType,
      Path symbolsOut,
      AndroidCompiledDataDeserializer deserializer,
      boolean throwOnResourceConflict,
      ExecutorServiceCloser executorService)
      throws IOException {
    AndroidDataMerger merger =
        AndroidDataMerger.createWithPathDeduplictor(
            executorService, deserializer, AndroidDataMerger.NoopSourceChecker.create());
    final UnwrittenMergedAndroidData merged =
        merger.loadAndMerge(
            transitive,
            direct,
            primary,
            manifest,
            packageType.equals(VariantType.DEFAULT),
            throwOnResourceConflict);
    AndroidDataSerializer serializer = AndroidDataSerializer.create();
    merged.serializeTo(serializer);
    serializer.flushTo(symbolsOut);
    return symbolsOut;
  }

  /** Thrown when there is a unexpected condition during merging. */
  public static class MergingException extends UserException {

    private MergingException(Throwable e) {
      super("Error during merging", e);
    }

    private MergingException(String message) {
      super("Merging Error: \n" + message);
    }

    static MergingException wrapException(Throwable e) {
      return new MergingException(e);
    }

    static MergingException withMessage(String message) {
      return new MergingException(message);
    }
  }

  static final Logger logger = Logger.getLogger(AndroidResourceProcessor.class.getName());

  /**
   * Merges all secondary resources with the primary resources, given that the primary resources
   * have not yet been parsed and serialized.
   */
  public static MergedAndroidData mergeDataAndWrite(
      final UnvalidatedAndroidData primary,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut,
      final List<String> filteredResources,
      boolean throwOnResourceConflict) {
    try (ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {
      final ParsedAndroidData parsedPrimary = ParsedAndroidData.from(primary);
      return mergeDataAndWrite(
          parsedPrimary,
          primary.getManifest(),
          direct,
          transitive,
          resourcesOut,
          assetsOut,
          cruncher,
          type,
          symbolsOut,
          /* rclassWriter= */ null,
          AndroidParsedDataDeserializer.withFilteredResources(filteredResources),
          throwOnResourceConflict,
          executorService);
    } catch (IOException e) {
      throw MergingException.wrapException(e);
    }
  }

  /**
   * Merges all secondary resources with the primary resources, given that the primary resources
   * have been separately parsed and serialized.
   */
  public static MergedAndroidData mergeDataAndWrite(
      final SerializedAndroidData primary,
      final Path primaryManifest,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      final Path resourcesOut,
      final Path assetsOut,
      @Nullable final PngCruncher cruncher,
      final VariantType type,
      @Nullable final Path symbolsOut,
      @Nullable final AndroidResourceClassWriter rclassWriter,
      boolean throwOnResourceConflict,
      ListeningExecutorService executorService) {
    final ParsedAndroidData.Builder primaryBuilder = ParsedAndroidData.Builder.newBuilder();
    final AndroidParsedDataDeserializer deserializer = AndroidParsedDataDeserializer.create();
    primary.deserialize(
        DependencyInfo.DependencyType.PRIMARY, deserializer, primaryBuilder.consumers());
    ParsedAndroidData primaryData = primaryBuilder.build();
    return mergeDataAndWrite(
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
        deserializer,
        throwOnResourceConflict,
        executorService);
  }

  /** Merges all secondary resources with the primary resources. */
  private static MergedAndroidData mergeDataAndWrite(
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
      AndroidParsedDataDeserializer deserializer,
      boolean throwOnResourceConflict,
      ListeningExecutorService executorService) {
    Stopwatch timer = Stopwatch.createStarted();
    try {
      UnwrittenMergedAndroidData merged =
          mergeData(
              executorService,
              transitive,
              direct,
              primary,
              primaryManifest,
              type != VariantType.LIBRARY,
              deserializer,
              throwOnResourceConflict,
              ContentComparingChecker.create());
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
      throw MergingException.wrapException(e);
    } finally {
      logger.fine(
          String.format("write merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }

  private static UnwrittenMergedAndroidData mergeData(
      ListeningExecutorService executorService,
      List<? extends SerializedAndroidData> transitive,
      List<? extends SerializedAndroidData> direct,
      ParsedAndroidData primary,
      Path primaryManifest,
      boolean allowPrimaryOverrideAll,
      AndroidDataDeserializer deserializer,
      boolean throwOnResourceConflict,
      SourceChecker checker) {
    Stopwatch timer = Stopwatch.createStarted();
    // TODO(b/74333698): Always check the contents of conflicting resources
    try {
      AndroidDataMerger merger =
          AndroidDataMerger.createWithPathDeduplictor(executorService, deserializer, checker);
      return merger.loadAndMerge(
          transitive,
          direct,
          primary,
          primaryManifest,
          allowPrimaryOverrideAll,
          throwOnResourceConflict);
    } finally {
      logger.fine(String.format("merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }

  /**
   * Merges all secondary compiled resources with the primary compiled resources, given that the
   * primary resources have been separately compiled
   */
  static void mergeCompiledData(
      final SerializedAndroidData primary,
      final Path primaryManifest,
      final List<? extends SerializedAndroidData> direct,
      final List<? extends SerializedAndroidData> transitive,
      @Nullable final AndroidResourceClassWriter rclassWriter,
      @Nullable PlaceholderRTxtWriter rTxtWriter,
      boolean throwOnResourceConflict,
      ListeningExecutorService executorService) {
    final ParsedAndroidData.Builder primaryBuilder = ParsedAndroidData.Builder.newBuilder();
    final AndroidDataDeserializer deserializer =
        AndroidCompiledDataDeserializer.create(/*includeFileContentsForValidation=*/ true);
    primary.deserialize(
        DependencyInfo.DependencyType.PRIMARY, deserializer, primaryBuilder.consumers());
    ParsedAndroidData primaryData = primaryBuilder.build();
    Stopwatch timer = Stopwatch.createStarted();
    try {
      UnwrittenMergedAndroidData merged =
          mergeData(
              executorService,
              transitive,
              direct,
              primaryData,
              primaryManifest,
              false,
              deserializer,
              throwOnResourceConflict,
              AndroidDataMerger.NoopSourceChecker.create());
      timer.reset().start();
      merged.writeResourceClass(rclassWriter);
      if (rTxtWriter != null) {
        merged.writeRTxt(rTxtWriter);
      }
      logger.fine(
          String.format("write classes finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      timer.reset().start();
    } catch (IOException e) {
      throw MergingException.wrapException(e);
    } finally {
      logger.fine(
          String.format("write merge finished in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }
}
