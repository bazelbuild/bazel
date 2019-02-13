// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildeventstream;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.ListenableFuture;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Transport interface for the build event protocol.
 *
 * <p>All implementations need to be thread-safe. All methods are expected to return quickly.
 *
 * <p>Notice that this interface does not provide any error handling API. A transport may choose
 * to log interesting errors to the command line and/or abort the whole build.
 */
@ThreadSafe
public interface BuildEventTransport {
  /**
   * The name of this transport as can be displayed to a user.
   */
  String name();

  /**
   * Writes a build event to an endpoint. This method will always return quickly and will not wait
   * for the write to complete.
   *
   * <p>In case the transport is in error, this method still needs to be able to accept build
   * events. It may choose to ignore them, though.
   *
   * @param event the event to sendBuildEvent.
   */
  void sendBuildEvent(BuildEvent event);

  /**
   * Initiates a close. Callers may listen to the returned future to be notified when the close is
   * complete i.e. wait for all build events to be sent. The future may contain any information
   * about possible transport errors.
   *
   * <p>This method might be called multiple times without any effect after the first call.
   *
   * <p>This method should not throw any exceptions.
   */
  ListenableFuture<Void> close();

  @VisibleForTesting
  @Nullable
  BuildEventArtifactUploader getUploader();
}
