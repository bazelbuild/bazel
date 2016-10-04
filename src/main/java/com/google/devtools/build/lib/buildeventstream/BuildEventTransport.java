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

import java.io.IOException;

/** Interface for transports of the build-event stream. */
public interface BuildEventTransport {

  /**
   * Ensure that the event will eventually be reported to the receiver this object is a transport
   * for; the transport is responsible that events arrive at the endpoint in the order they are sent
   * by invocations of this method.
   */
  void sendBuildEvent(BuildEvent event) throws IOException;

  /**
   * Close all open resources, if any. This method will be called on the transport after all events
   * have been sent. If a transport is stateless, it is correct to do nothing.
   */
  void close() throws IOException;
}
