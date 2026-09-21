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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventservice.client.CommandContext;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** Implementation of {@link CommandContext}. */
public record CommandContextImpl(
    String buildId,
    String invocationId,
    int attemptNumber,
    Set<String> keywords,
    @Nullable String projectId,
    boolean checkPrecedingLifecycleEvents,
    List<byte[]> streamMetadata)
    implements CommandContext {

  public CommandContextImpl {
    checkNotNull(buildId, "buildId");
    checkNotNull(invocationId, "invocationId");
    checkNotNull(keywords, "keywords");
    checkArgument(attemptNumber >= 1, "attemptNumber must be >= 1");
    streamMetadata = ImmutableList.copyOf(streamMetadata);
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link CommandContextImpl}. */
  public static final class Builder {
    private String buildId;
    private String invocationId;
    private int attemptNumber;
    private Set<String> keywords;
    private String projectId;
    private boolean checkPrecedingLifecycleEvents;
    private List<byte[]> streamMetadata = ImmutableList.of();

    private Builder() {}

    @CanIgnoreReturnValue
    public Builder setBuildId(String buildId) {
      this.buildId = buildId;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setInvocationId(String invocationId) {
      this.invocationId = invocationId;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setAttemptNumber(int attemptNumber) {
      this.attemptNumber = attemptNumber;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setKeywords(Set<String> keywords) {
      this.keywords = keywords;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setProjectId(@Nullable String projectId) {
      this.projectId = projectId;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setCheckPrecedingLifecycleEvents(boolean checkPrecedingLifecycleEvents) {
      this.checkPrecedingLifecycleEvents = checkPrecedingLifecycleEvents;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setStreamMetadata(List<byte[]> streamMetadata) {
      this.streamMetadata = streamMetadata;
      return this;
    }

    public CommandContextImpl build() {
      return new CommandContextImpl(
          buildId,
          invocationId,
          attemptNumber,
          keywords,
          projectId,
          checkPrecedingLifecycleEvents,
          streamMetadata);
    }
  }
}
