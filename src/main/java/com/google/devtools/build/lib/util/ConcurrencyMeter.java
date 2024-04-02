// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.clock.Clock;
import com.google.errorprone.annotations.ThreadSafe;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import java.time.Instant;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A dispenser for up to 'total' simultaneous units of some resource. The resource itself is not
 * accessed through this object; this is basically an asynchronous interface to a counting
 * semaphore.
 */
@ThreadSafe
public final class ConcurrencyMeter {

  private final String name;
  private final long total;

  @GuardedBy("this")
  private final Clock clock;

  @GuardedBy("this")
  private long leased = 0;

  @GuardedBy("this")
  private final Queue<PendingJob> queue = new PriorityQueue<>();

  @GuardedBy("this")
  private long maxLeased = 0;

  @GuardedBy("this")
  private Instant maxLeasedTimestamp;

  /**
   * Creates a meter with the given number of units.
   *
   * @param name an identifier for this meter, for use in {@link #getStats}
   * @param total total number of permits that may be dispensed
   * @param clock provides the current time for {@link Stats#maxLeasedTimeMs()}
   */
  public ConcurrencyMeter(String name, long total, Clock clock) {
    this.name = checkNotNull(name);
    this.total = total;
    this.clock = checkNotNull(clock);
  }

  @VisibleForTesting
  synchronized int queueSize() {
    return queue.size();
  }

  /**
   * Enqueues a request for {@code quantity} units of the resource managed by this meter. When the
   * request is filled, the result becomes available.
   *
   * <p>The resource must be released either by cancelling the future or by calling {@link
   * Ticket#done} on the ticket after the future completes.
   *
   * @param quantity number of units of resources to acquire
   * @param priority requests with greater priority complete earlier
   * @return a future which grants resources only when it completes successfully
   */
  public ListenableFuture<Ticket> request(long quantity, long priority) {
    checkArgument(quantity >= 0);
    PendingJob job = new PendingJob(quantity, priority);
    synchronized (this) {
      queue.add(job);
    }
    schedule();
    return job.futureTicket;
  }

  /** Statistics about a ConcurrencyMeter. */
  public record Stats(
      String name, long total, long leased, long maxLeased, long maxLeasedTimeMs) {}

  public synchronized Stats getStats() {
    return new Stats(
        name, total, leased, maxLeased, maxLeased > 0 ? maxLeasedTimestamp.toEpochMilli() : 0);
  }

  private synchronized void release(long quantity) {
    checkState(leased >= quantity, "quantity (%s) > leased (%s)", quantity, leased);
    leased -= quantity;
  }

  private void releaseAndSchedule(long quantity) {
    release(quantity);
    schedule();
  }

  private void schedule() {
    while (true) {
      PendingJob job;
      synchronized (this) {
        job = queue.peek();
        if (job == null || (leased + job.quantity > total && leased > 0)) {
          return;
        }
        queue.remove();
        leased += job.quantity;

        if (leased >= maxLeased) {
          maxLeased = leased;
          maxLeasedTimestamp = clock.now();
        }
      }

      // Set the future outside synchronized block to avoid holding the lock when executing future's
      // callbacks which may hold other locks and call into ConcurrencyMeter causing deadlocks.
      // See: b/319411390
      if (!job.futureTicket.set(new ReleasingTicket(job.quantity))) {
        // The future may have been cancelled. Release immediately. If the build was interrupted, we
        // may encounter a long chain of cancelled tickets - avoid calling ticket.done() or
        // releaseAndSchedule() which would process them recursively.
        release(job.quantity);
      }
    }
  }

  private final class ReleasingTicket implements Ticket {
    private final long quantity;
    private final AtomicBoolean released = new AtomicBoolean(false);

    ReleasingTicket(long quantity) {
      this.quantity = quantity;
    }

    @Override
    public void done() {
      boolean alreadyReleased = released.getAndSet(true);
      checkState(!alreadyReleased, "Already released %s units", quantity);
      releaseAndSchedule(quantity);
    }
  }

  private static final class PendingJob implements Comparable<PendingJob> {
    private final SettableFuture<Ticket> futureTicket = SettableFuture.create();
    private final long quantity;
    private final long priority;

    PendingJob(long quantity, long priority) {
      this.quantity = quantity;
      this.priority = priority;
    }

    @Override
    public int compareTo(PendingJob o) {
      return Long.compare(o.priority, priority);
    }
  }

  /** A ticket denoting resource acquisition. */
  public interface Ticket {
    /** Releases the associated resources. Must be called exactly once. */
    void done();
  }
}
