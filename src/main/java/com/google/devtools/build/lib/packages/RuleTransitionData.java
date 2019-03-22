package com.google.devtools.build.lib.packages;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;

/**
 * Helper class which contains data used by a {@link TransitionFactory} to create a transition for
 * rules and attributes.
 */
// This class is in lib.packages in order to access AttributeMap, which is not available to
// the lib.analysis.config.transitions package.
@AutoValue
public abstract class RuleTransitionData implements TransitionFactory.TransitionFactoryData {
  /** Returns the {@link AttributeMap} which can be used to create a transition. */
  public abstract AttributeMap attributes();

  // TODO(https://github.com/bazelbuild/bazel/issues/7814): Add further data fields as needed by
  // transition factory instances.

  /** Returns a new {@link RuleTransitionData} instance. */
  public static RuleTransitionData create(AttributeMap attributes) {
    return new AutoValue_RuleTransitionData(attributes);
  }
}
