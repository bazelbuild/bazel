// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;
import javax.annotation.Nullable;

/**
 * {@link Subject} for {@link NodeEntry}. Please add to this class if you need more functionality!
 */
public class NodeEntrySubject extends Subject {
  private final NodeEntry actual;

  NodeEntrySubject(FailureMetadata failureMetadata, NodeEntry nodeEntry) {
    super(failureMetadata, nodeEntry);
    this.actual = nodeEntry;
  }

  public Subject hasVersionThat() {
    return check("getVersion()").that(actual.getVersion());
  }

  public IterableSubject hasTemporaryDirectDepsThat() {
    return check("getTemporaryDirectDeps()")
        .that(Iterables.concat(actual.getTemporaryDirectDeps()));
  }

  public Subject addReverseDepAndCheckIfDone(@Nullable SkyKey reverseDep)
      throws InterruptedException {
    return check("addReverseDepAndCheckIfDone()")
        .that(actual.addReverseDepAndCheckIfDone(reverseDep));
  }
}
