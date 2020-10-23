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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Collection;
import javax.annotation.Nullable;

/** A {@link BuildEvent} reporting an event not coming due to the build being aborted. */
public final class AbortedEvent extends GenericBuildEvent {

  private final AbortReason reason;
  private final String description;
  @Nullable private final Label label;

  public AbortedEvent(BuildEventId id, AbortReason reason, String description) {
    this(id, reason, description, /*label=*/ null);
  }

  public AbortedEvent(
      BuildEventId id, AbortReason reason, String description, @Nullable Label label) {
    this(id, /*children=*/ ImmutableList.of(), reason, description, label);
  }

  public AbortedEvent(
      BuildEventId id, Collection<BuildEventId> children, AbortReason reason, String description) {
    this(id, children, reason, description, /*label=*/ null);
  }

  private AbortedEvent(
      BuildEventId id,
      Collection<BuildEventId> children,
      AbortReason reason,
      String description,
      @Nullable Label label) {
    super(id, children);
    this.reason = reason;
    this.description = description;
    this.label = label;
  }

  @Nullable
  public Label getLabel() {
    return label;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this)
        .setAborted(Aborted.newBuilder().setReason(reason).setDescription(description).build())
        .build();
  }
}
