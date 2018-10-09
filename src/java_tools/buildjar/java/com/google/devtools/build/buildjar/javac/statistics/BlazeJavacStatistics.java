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
import com.google.errorprone.annotations.MustBeClosed;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Context.Factory;
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
      context.put(
          Builder.class,
          (Factory<Builder>)
              c -> {
                Builder instance = newBuilder();
                c.put(Builder.class, instance);
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
    return new AutoValue_BlazeJavacStatistics.Builder();
  }

  public abstract ImmutableListMultimap<TickKey, Duration> timingTicks();

  public abstract ImmutableListMultimap<String, Duration> errorProneTicks();

  // TODO(glorioso): We really need to think out more about what data to collect/store here.

  /** Known sources of timing information */
  public enum TickKey {
    DAGGER,
  }

  /**
   * Builder of {@link BlazeJavacStatistics} instances.
   *
   * <p>Normally available through a {@link Context} via: {@code context.getKey({@link
   * BlazeJavacStatistics.Builder}.class} after {@link BlazeJavacStatistics#preRegister(Context)}
   * has been called.
   */
  @AutoValue.Builder
  public abstract static class Builder {

    abstract ImmutableListMultimap.Builder<TickKey, Duration> timingTicksBuilder();

    abstract ImmutableListMultimap.Builder<String, Duration> errorProneTicksBuilder();

    public Builder addErrorProneTiming(String key, Duration value) {
      errorProneTicksBuilder().put(key, value);
      return this;
    }

    public abstract BlazeJavacStatistics build();

    public Builder addTick(TickKey key, Duration elapsed) {
      timingTicksBuilder().put(key, elapsed);
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
