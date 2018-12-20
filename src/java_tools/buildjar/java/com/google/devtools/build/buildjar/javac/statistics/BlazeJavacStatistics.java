// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.buildjar.javac.statistics;

import com.google.auto.value.AutoValue;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.MustBeClosed;
import com.sun.tools.javac.util.Context;
import java.time.Duration;
import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.Consumer;

/**
 * A class representing statistics for an invocation of {@link
 * com.google.devtools.build.buildjar.javac.BlazeJavacMain#compile}.
 *
 * <p>This will generally include performance statistics (how long the process ran, how many times
 * did an annotation processor run, how many Error Prone checks were checked, etc.).
 */
@AutoValue
public abstract class BlazeJavacStatistics {

  // Weak refs to contexts we've init'ed into
  private static final Set<Context> contextsInitialized =
      Collections.newSetFromMap(new WeakHashMap<>());

  public static void preRegister(Context context) {
    if (contextsInitialized.add(context)) {
      context.<Builder>put(
          Builder.class,
          ctx -> {
            Builder instance = newBuilder();
            ctx.put(Builder.class, instance);
            return instance;
          });
    } else {
      throw new IllegalStateException("Initialize called twice!");
    }
  }

  public static BlazeJavacStatistics empty() {
    return newBuilder().build();
  }

  private static Builder newBuilder() {
    return new AutoValue_BlazeJavacStatistics.Builder()
        .transitiveClasspathLength(0)
        .reducedClasspathLength(0)
        .minClasspathLength(0)
        .transitiveClasspathFallback(false);
  }

  public abstract ImmutableMap<AuxiliaryDataSource, byte[]> auxiliaryData();

  @Deprecated // use auxiliaryData() instead.
  public abstract ImmutableListMultimap<TickKey, Duration> timingTicks();

  public abstract ImmutableListMultimap<String, Duration> errorProneTicks();

  public abstract ImmutableSet<String> processors();

  public abstract int transitiveClasspathLength();

  public abstract int reducedClasspathLength();

  public abstract int minClasspathLength();

  public abstract boolean transitiveClasspathFallback();

  // TODO(glorioso): We really need to think out more about what data to collect/store here.

  /**
   * Known sources of timing information
   *
   * @deprecated Instead, use {@link AuxiliaryDataSource}.
   */
  @Deprecated
  public enum TickKey {
    DAGGER,
  }

  /**
   * Known sources of additional data to add to the statistics. Each data source can put a single
   * byte[] of serialized proto data into this statistics object with {@link
   * Builder#addAuxiliaryData}
   */
  public enum AuxiliaryDataSource {
    DAGGER,
  }

  public abstract Builder toBuilder();

  /**
   * Builder of {@link BlazeJavacStatistics} instances.
   *
   * <p>Normally available through a {@link Context} via: {@code context.getKey({@link
   * BlazeJavacStatistics.Builder}.class} after {@link BlazeJavacStatistics#preRegister(Context)}
   * has been called.
   */
  @AutoValue.Builder
  public abstract static class Builder {

    @Deprecated // use auxiliaryDataBuilder() instead
    abstract ImmutableListMultimap.Builder<TickKey, Duration> timingTicksBuilder();

    abstract ImmutableListMultimap.Builder<String, Duration> errorProneTicksBuilder();

    abstract ImmutableMap.Builder<AuxiliaryDataSource, byte[]> auxiliaryDataBuilder();

    abstract ImmutableSet.Builder<String> processorsBuilder();

    public abstract Builder transitiveClasspathLength(int length);

    public abstract Builder reducedClasspathLength(int length);

    public abstract Builder minClasspathLength(int length);

    public abstract Builder transitiveClasspathFallback(boolean fallback);

    public Builder addErrorProneTiming(String key, Duration value) {
      errorProneTicksBuilder().put(key, value);
      return this;
    }

    public abstract BlazeJavacStatistics build();

    @Deprecated // use addAuxiliaryData(key, byte[]) with a serialized Any proto message.
    public Builder addTick(TickKey key, Duration elapsed) {
      timingTicksBuilder().put(key, elapsed);
      return this;
    }

    /**
     * Add an auxiliary attachment of data to this statistics object. The data should be a proto
     * serialization of a google.protobuf.Any protobuf.
     *
     * <p>Since this method is called across the boundaries of an annotation processorpath and the
     * runtime classpath of the compiler, we want to reduce the number of classes mentioned, hence
     * the byte[] data type. If we find a way to make this more safe, we would prefer to use a
     * protobuf ByteString instead for its immutability.
     */
    public Builder addAuxiliaryData(AuxiliaryDataSource key, byte[] serializedData) {
      auxiliaryDataBuilder().put(key, serializedData.clone());
      return this;
    }

    @MustBeClosed
    public final StopwatchSpan newTimingSpan(Consumer<Duration> consumer) {
      Stopwatch stopwatch = Stopwatch.createStarted();
      return () -> {
        stopwatch.stop();
        consumer.accept(stopwatch.elapsed());
      };
    }

    public Builder addProcessor(String processor) {
      processorsBuilder().add(processor);
      return this;
    }
  }

  /**
   * A simple AutoClosable interface where the {@link #close} method doesn't throw an exception.
   *
   * <p>Returned from {@link Builder#newTimingSpan(Consumer)}
   */
  public interface StopwatchSpan extends AutoCloseable {
    @Override
    void close();
  }
}
