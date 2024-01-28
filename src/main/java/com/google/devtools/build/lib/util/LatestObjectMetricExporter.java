// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.lang.ref.Reference;
import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.util.function.Supplier;
import javax.annotation.concurrent.GuardedBy;

/**
 * Exporter for a callback metric instrumenting a singleton that may not be created, and when
 * created, may be discarded and re-created.
 *
 * <p>Lazily registers a callback-metric with a thread-safe {@link Supplier} of the latest value of
 * that reference. Lazily registering the callback metric reduces metric pollution when the
 * instrumented codepaths are never executed.
 *
 * <p>Weak/soft references must be used to allow the instrumented object to be GCed; callbacks must
 * expect {@code null} values. Note that in some instrumentation libraries it is impossible to stop
 * exporting a given metric.
 *
 * <p>Simple usage example based on the open-source {@code io.opentelemetry.api.metrics} API:
 *
 * <pre>
 *   class FooManager {
 *     private static final ObservableLongMeasurement fooMetric =
 *         MyMeterProvider.get().gaugeBuilder("foo").ofLongs().buildObserver();
 *     private static final ObservableLongMeasurement barMetric =
 *         MyMeterProvider.get().gaugeBuilder("bar").ofLongs().buildObserver();
 *     static void updateMetric(FooManager manager) {
 *       fooMetric.record(manager == null ? 0L : manager.getFoo());
 *       barMetric.record(manager == null ? 0L : manager.getBar());
 *     }
 *     private static final LatestObjectMetricExporter&lt;FooManager&gt; FOO_MANAGER_EXPORTER =
 *         new LatestObjectMetricExporter&lt;&gt;(
 *             LatestObjectMetricExporter.Strength.WEAK,
 *             (supplier) -> MyMeterProvider.get().batchCallback(
 *                 () -> updateMetric(supplier.get()),
 *                 fooMetric,
 *                 barMetric));
 *
 *     // Need some state fields to export.
 *     \@GuardedBy("this") private Foo foo;
 *     \@GuardedBy("this") private Bar bar;
 *     FooManager(Foo foo, Bar bar) {
 *       // Initialize state fields before exporting the FooManager.
 *       this.bar = bar;
 *       this.bar = bar;
 *       FOO_MANAGER_EXPORTER.setLatestInstance(this);
 *     }
 *     // Measurements must be thread-safe.
 *     synchronized long getFoo() {
 *       return bar.getFooSize();
 *     }
 *     synchronized long getBar() {
 *       return bar.getBarSize();
 *     }
 *   }
 * </pre>
 *
 * @param <T> Type of the <em>latest object</em> being tracked.
 */
@ThreadSafe
public final class LatestObjectMetricExporter<T> {

  /**
   * Metric-specific callback, run once the first time a {@link LatestObjectMetricExporter} is used.
   */
  public interface CallbackRegistration<T> {
    /**
     * One-time setup method expected to register callback metrics with the instrumentation
     * library's metric registry.
     *
     * <p>Callbacks are expected to use the given {@link Supplier} to get the latest instance (or
     * {@code null} if the latest instance has been GCed).
     */
    void register(Supplier<T> refSupplier);
  }

  /** Kind of reference held by the exporter. */
  public enum Strength {
    /** Creates {@link WeakReference} instances. */
    WEAK,
    /** Creates {@link SoftReference} instances. */
    SOFT;

    /** Create a new Reference for the given value, which may be {@code null}. */
    <T> Reference<T> makeRef(T value) {
      switch (this) {
        case WEAK:
          return new WeakReference<>(value);
        case SOFT:
          return new SoftReference<>(value);
      }
      throw new IllegalStateException("unexpected reference strength: " + name());
    }
  }

  /** The reference strength used for the latest object. */
  private final Strength strength;

  /**
   * Registration callback that will be invoked at most once, the first time {@link
   * LatestObjectMetricExporter#setLatestInstance(T)} is called.
   */
  private final CallbackRegistration<T> registration;

  /** Flag that is set after the callback registration method has been called. */
  @GuardedBy("this")
  private boolean callbackRegistered = false;

  /**
   * Reference to the last {@link T object} created by Blaze; as a weak/soft reference, will be null
   * if it has been GCed.
   *
   * <p>We don't use an {@link java.util.concurrent.atomic.AtomicReference} because we don't know
   * (other than a finalizer) when to clear the reference to avoid leaking memory.
   */
  @GuardedBy("this")
  private Reference<T> reference;

  /** Create a singleton exporter with the given reference strength and registration callback. */
  public LatestObjectMetricExporter(Strength strength, CallbackRegistration<T> registration) {
    this.strength = strength;
    this.registration = registration;
    reference = strength.makeRef(null);
  }

  /**
   * Sets the latest instance of the instrumented singleton (through the Supplier passed to the
   * exporter's {@link CallbackRegistration}).
   *
   * <p>If this is the first time the method has been called, {@code registration#register()} will
   * be called after changing {@link #reference}.
   */
  public synchronized void setLatestInstance(T value) {
    reference = strength.makeRef(value);
    if (!callbackRegistered) {
      registration.register(
          () -> {
            synchronized (LatestObjectMetricExporter.this) {
              return reference.get();
            }
          });
      callbackRegistered = true;
    }
  }
}
