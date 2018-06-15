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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Set;

/**
 * Interface for objects that can be posted on the public event stream.
 *
 * <p>Objects posted on the build-event stream will implement this interface. This allows
 * pass-through of events, as well as proper chaining of events.
 */
public interface BuildEvent extends ChainableEvent, ExtendedEventHandler.Postable {

  /**
   * Returns a list of files that are referenced in the protobuf representation returned by {@link
   * #asStreamProto(BuildEventContext)}.
   *
   * <p>This method is different from {@code EventReportingArtifacts#reportedArtifacts()} in that it
   * only returns files directly referenced in the protobuf returned by {@link
   * #asStreamProto(BuildEventContext)}.
   *
   * <p>Note the consistency requirement - you must not attempt to pass Path objects to the
   * {@link PathConverter} unless you have returned the Path object here.
   */
  // TODO(ulfjack): Consider moving the upload call to the BuildEventContext and returning a map
  // from Path to URI, rather than a callback.
  default Set<Path> referencedLocalFiles() {
    return ImmutableSet.of();
  }

  /**
   * Provide a binary representation of the event.
   *
   * <p>Provide a presentation of the event according to the specified binary format, as appropriate
   * protocol buffer.
   */
  BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context);
}
