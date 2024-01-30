// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.InconsistencyStats;
import java.util.Collection;
import javax.annotation.Nullable;

/** {@link GraphInconsistencyReceiver} that forwards all operations to a delegate. */
public final class DelegatingGraphInconsistencyReceiver implements GraphInconsistencyReceiver {
  private GraphInconsistencyReceiver delegate;

  public DelegatingGraphInconsistencyReceiver(GraphInconsistencyReceiver delegate) {
    this.delegate = checkNotNull(delegate);
  }

  public void setDelegate(GraphInconsistencyReceiver delegate) {
    this.delegate = checkNotNull(delegate);
  }

  @Override
  public void noteInconsistencyAndMaybeThrow(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    delegate.noteInconsistencyAndMaybeThrow(key, otherKeys, inconsistency);
  }

  @Override
  public InconsistencyStats getInconsistencyStats() {
    return delegate.getInconsistencyStats();
  }

  @Override
  public void reset() {
    delegate.reset();
  }
}
