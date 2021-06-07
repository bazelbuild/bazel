package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/**
 * Notifies that an in-flight action is making progress..
 */
public class ActionProgressEvent implements ProgressLike {
  private final ActionExecutionMetadata action;
  private final String progressId;
  private final String progress;
  private final boolean isFinished;

  public ActionProgressEvent(ActionExecutionMetadata action, String progressId, String progress, boolean isFinished) {
    this.action = action;
    this.progressId = progressId;
    this.progress = progress;
    this.isFinished = isFinished;
  }

  /** Gets the metadata associated with the action being scheduled. */
  public ActionExecutionMetadata getActionMetadata() {
    return action;
  }

  /**
   * The id that uniquely determines the progress among all progress events within an action.
   */
  public String getProgressId() {
    return progressId;
  }

  /** Human readable description of the progress */
  public String getProgress() {
    return progress;
  }

  /** Whether the download progress reported about is finished already */
  public boolean isFinished() {
    return isFinished;
  }
}
