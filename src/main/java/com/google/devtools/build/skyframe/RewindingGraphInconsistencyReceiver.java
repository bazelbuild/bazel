package com.google.devtools.build.skyframe;

import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * A {@link GraphInconsistencyReceiver} which allows rewinding to re-perform actions whose outputs
 * have been lost.
 */
public class RewindingGraphInconsistencyReceiver implements GraphInconsistencyReceiver {

  @Override
  public void noteInconsistencyAndMaybeThrow(SkyKey key, @Nullable Collection<SkyKey> otherKeys,
      Inconsistency inconsistency) {
    if (Inconsistency.PARENT_FORCE_REBUILD_OF_CHILD.equals(inconsistency) || Inconsistency.RESET_REQUESTED.equals(inconsistency)) {
      return;
    }
    throw new IllegalStateException(
        "Unexpected inconsistency: " + key + ", " + otherKeys + ", " + inconsistency);
  }

  @Override
  public boolean restartPermitted() {
    return true;
  }
}
