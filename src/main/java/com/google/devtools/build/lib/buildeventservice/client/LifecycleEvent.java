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

package com.google.devtools.build.lib.buildeventservice.client;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.time.Instant;

/** A lifecycle event. */
@SkybridgeInterface
public interface LifecycleEvent {
  /** The time at which the event occurred. */
  Instant eventTime();

  /** The status of an invocation. */
  enum InvocationStatus {
    /** No information is available about the invocation status. */
    UNKNOWN,
    /** The invocation succeeded. */
    SUCCEEDED,
    /** The invocation failed. */
    FAILED,
  }

  /** The lifecycle event signalling that the build was enqueued. */
  interface BuildEnqueued extends LifecycleEvent {}

  /** The lifecycle event signalling that the invocation was started. */
  interface InvocationStarted extends LifecycleEvent {}

  /** The lifecycle event signalling that the invocation was finished. */
  interface InvocationFinished extends LifecycleEvent {
    /** The invocation status. */
    InvocationStatus status();
  }

  /** The lifecycle event signalling that the build was finished. */
  interface BuildFinished extends LifecycleEvent {
    /** The invocation status. */
    InvocationStatus status();
  }
}
