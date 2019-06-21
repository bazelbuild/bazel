// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.repository;

import static com.google.devtools.build.lib.cmdline.LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;
import java.util.Collection;

/**
 * Event indicating that a failure is related to a given external repository; this is in particular
 * the case, if fetching that repository failed.
 */
public class RepositoryFailedEvent implements ProgressLike, BuildEvent {
  private final String repo;
  private final String message;

  public RepositoryFailedEvent(String repo, String message) {
    this.repo = repo;
    this.message = message;
  }

  public String getRepo() {
    return repo;
  }

  @Override
  public BuildEventId getEventId() {
    String strippedRepoName = repo;
    if (strippedRepoName.startsWith("@")) {
      strippedRepoName = strippedRepoName.substring(1);
    }
    try {
      Label label = Label.create(EXTERNAL_PACKAGE_IDENTIFIER, strippedRepoName);
      return BuildEventId.unconfiguredLabelId(label);
    } catch (LabelSyntaxException e) {
      // As the repository name was accepted earlier, the label construction really shouldn't fail.
      // In any case, return something still referring to the repository.
      return BuildEventId.unknownBuildEventId(EXTERNAL_PACKAGE_IDENTIFIER + ":" + strippedRepoName);
    }
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.<BuildEventId>of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context) {
    return GenericBuildEvent.protoChaining(this)
        .setAborted(
            BuildEventStreamProtos.Aborted.newBuilder()
                .setReason(BuildEventStreamProtos.Aborted.AbortReason.LOADING_FAILURE)
                .setDescription(message)
                .build())
        .build();
  }
}
