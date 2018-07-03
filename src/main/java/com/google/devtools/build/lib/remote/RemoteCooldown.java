package com.google.devtools.build.lib.remote;

import static com.google.devtools.build.lib.remote.Retrier.CircuitBreaker.State.*;

import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import javax.annotation.concurrent.ThreadSafe;

@ThreadSafe
class RemoteCooldown implements CircuitBreaker {
  private final int delay; // seconds
  private final int minAcceptSuccess;

  private Instant offUntil = Instant.MIN;
  private int successSinceOff;

  RemoteCooldown(int delay, int minAcceptSuccess) {
    this.delay = delay;
    this.minAcceptSuccess = minAcceptSuccess;

    successSinceOff = minAcceptSuccess; // start off optimistic
  }

  @Override
  public synchronized State state() {
    if (isOff(Instant.now())) {
      return REJECT_CALLS;
    }
    return successSinceOff >= minAcceptSuccess ? ACCEPT_CALLS : TRIAL_CALL;
  }

  @Override
  public void recordFailure() {
    // no failure based reaction
  }

  @Override
  public synchronized void recordSuccess() {
    if (successSinceOff < minAcceptSuccess) {
      successSinceOff++;
    }
  }

  private boolean isOff(Instant at) {
    return at.compareTo(offUntil) < 0;
  }

  synchronized void start() {
    Instant now = Instant.now();
    if (!isOff(now)) {
      offUntil = now.plus(delay, ChronoUnit.SECONDS);
    }
    successSinceOff = 0;
  }
}
