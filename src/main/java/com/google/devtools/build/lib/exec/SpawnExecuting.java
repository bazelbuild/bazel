package com.google.devtools.build.lib.exec;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;

/**
 * Resources are acquired, and there was probably no cache hit. This MUST be posted before
 * attempting to execute the subprocess.
 *
 * <p>Caching {@link SpawnRunner} implementations should only post this after a failed cache
 * lookup, but may post this if cache lookup and execution happen within the same step, e.g. as
 * part of a single RPC call with no mechanism to report cache misses.
 */
@AutoValue
public abstract class SpawnExecuting implements ProgressStatus {
  public static SpawnExecuting create(String name) {
    return new AutoValue_SpawnExecuting(name);
  }

  public abstract String name();

  @Override
  public void postTo(ExtendedEventHandler eventHandler, ActionExecutionMetadata action) {
    eventHandler.post(new RunningActionEvent(action, name()));
  }
}
