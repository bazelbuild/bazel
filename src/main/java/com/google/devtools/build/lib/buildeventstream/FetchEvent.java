// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.Collection;

/**
 * A {@link BuildEvent} reporting that an external resource was fetched.
 *
 * <p>Events of this class will only be generated in builds that do the actual fetch, not in ones
 * that use a cached copy of the resource to download. In way, these events allow keeping track of
 * the access of external resources.
 */
public final class FetchEvent implements BuildEvent, ExtendedEventHandler.ProgressLike {
  private final String url;
  private final boolean success;

  public FetchEvent(String url, boolean success) {
    this.url = url;
    this.success = success;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.fetchId(url);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.<BuildEventId>of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.Fetch fetch =
        BuildEventStreamProtos.Fetch.newBuilder().setSuccess(success).build();
    return GenericBuildEvent.protoChaining(this).setFetch(fetch).build();
  }
}
