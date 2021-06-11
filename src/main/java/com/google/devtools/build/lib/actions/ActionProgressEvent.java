package com.google.devtools.build.lib.actions;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/** Notifies that an in-flight action is making progress. */
@AutoValue
public abstract class ActionProgressEvent implements ProgressLike {

  public static ActionProgressEvent create(
      ActionExecutionMetadata action, String progressId, String progress, boolean finished) {
    return new AutoValue_ActionProgressEvent(action, progressId, progress, finished);
  }

  /** Gets the metadata associated with the action being scheduled. */
  public abstract ActionExecutionMetadata action();

  /** The id that uniquely determines the progress among all progress events within an action. */
  public abstract String progressId();

  /** Human readable description of the progress */
  public abstract String progress();

  /** Whether the download progress reported about is finished already */
  public abstract boolean finished();
}
