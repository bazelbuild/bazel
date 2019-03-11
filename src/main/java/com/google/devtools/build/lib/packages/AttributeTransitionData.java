package com.google.devtools.build.lib.packages;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;

@AutoValue
public abstract class AttributeTransitionData extends TransitionFactoryData {
  public abstract AttributeMap attributes();

  public static AttributeTransitionData create(AttributeMap attributes) {
    return new AutoValue_AttributeTransitionData(attributes);
  }
}
