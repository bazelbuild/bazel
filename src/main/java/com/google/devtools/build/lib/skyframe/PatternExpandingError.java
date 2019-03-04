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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.util.Collection;
import java.util.List;

/** Event reporting about failure to expand a target pattern properly. */
public final class PatternExpandingError implements BuildEvent, BuildEventWithOrderConstraint {

  private final Collection<BuildEventId> postedAfter;
  private final List<String> pattern;
  private final String message;
  private final boolean skipped;

  private PatternExpandingError(List<String> allPatterns, List<String> pattern, String message, boolean skipped) {
    ImmutableList.Builder postedAfterBuilder = ImmutableList.builder();
    if (allPatterns.size() > pattern.size()) {
      postedAfterBuilder.add(BuildEventId.targetPatternExpanded(allPatterns));
    }
    this.postedAfter = postedAfterBuilder.build();
    this.pattern = pattern;
    this.message = message;
    this.skipped = skipped;
  }

  public List<String> getPattern() {
    return pattern;
  }

  public boolean getSkipped() {
    return skipped;
  }

  public static PatternExpandingError failed(List<String> pattern, String message) {
    return new PatternExpandingError(ImmutableList.of(), pattern, message, false);
  }

  public static PatternExpandingError failed(List<String> allPatterns, String term, String message) {
    return new PatternExpandingError(allPatterns, ImmutableList.of(term), message, false);
  }

  // This is unused right now - when we generate the error, we don't know if we're in keep_going
  // mode or not.
  public static PatternExpandingError skipped(String term, String message) {
    return new PatternExpandingError(ImmutableList.of(), ImmutableList.of(term), message, true);
  }

  @Override
  public BuildEventId getEventId() {
    if (skipped) {
      return BuildEventId.targetPatternSkipped(pattern);
    } else {
      return BuildEventId.targetPatternExpanded(pattern);
    }
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return postedAfter;
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.<BuildEventId>of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.Aborted failure =
        BuildEventStreamProtos.Aborted.newBuilder()
            .setReason(BuildEventStreamProtos.Aborted.AbortReason.LOADING_FAILURE)
            .setDescription(message)
            .build();
    return GenericBuildEvent.protoChaining(this).setAborted(failure).build();
  }
}
