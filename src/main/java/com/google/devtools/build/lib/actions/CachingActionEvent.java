package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/**
 * Notifies that an in-flight action is checking the cache.
 */
public class CachingActionEvent implements ProgressLike {

  private final ActionExecutionMetadata action;
  private final String strategy;

  /**
   * Constructs a new event.
   */
  public CachingActionEvent(ActionExecutionMetadata action, String strategy) {
    this.action = action;
    this.strategy = checkNotNull(strategy, "Strategy names are not optional");
  }

  /**
   * Gets the metadata associated with the action.
   */
  public ActionExecutionMetadata getActionMetadata() {
    return action;
  }

  /**
   * Gets the name of the strategy on which the action is caching.
   */
  public String getStrategy() {
    return strategy;
  }
}