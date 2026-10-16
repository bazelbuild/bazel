// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.buildeventservice.client.LifecycleEvent;
import java.time.Instant;

/** Implementation of {@link LifecycleEvent.InvocationFinished}. */
public record InvocationFinishedImpl(Instant eventTime, LifecycleEvent.InvocationStatus status)
    implements LifecycleEvent.InvocationFinished {
  public InvocationFinishedImpl {
    checkNotNull(eventTime, "eventTime");
    checkNotNull(status, "status");
  }
}
