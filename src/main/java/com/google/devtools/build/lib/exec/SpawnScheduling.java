package com.google.devtools.build.lib.exec;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.SchedulingActionEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;

/**
 * Spawn is waiting for local or remote resources to become available.
 */
@AutoValue
public abstract class SpawnScheduling implements ProgressStatus {
  public static SpawnScheduling create(String name) {
    return new AutoValue_SpawnScheduling(name);
  }

  public abstract String name();

  @Override
  public void postTo(ExtendedEventHandler eventHandler, ActionExecutionMetadata action) {
    eventHandler.post(new SchedulingActionEvent(action, name()));
  }
}
