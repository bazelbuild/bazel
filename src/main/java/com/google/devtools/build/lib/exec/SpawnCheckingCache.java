package com.google.devtools.build.lib.exec;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.CachingActionEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;

/** The {@link SpawnRunner} is looking for a cache hit. */
@AutoValue
public abstract class SpawnCheckingCache implements ProgressStatus {
  public static SpawnCheckingCache create(String name) {
    return new AutoValue_SpawnCheckingCache(name);
  }

  public abstract String name();

  @Override
  public void postTo(ExtendedEventHandler eventHandler, ActionExecutionMetadata action) {
    eventHandler.post(new CachingActionEvent(action, name()));
  }
}
