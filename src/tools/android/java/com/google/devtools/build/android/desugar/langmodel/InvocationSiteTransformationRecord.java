/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;

/** A record that tracks the method invocation transformations. */
@AutoValue
public abstract class InvocationSiteTransformationRecord {

  public abstract ImmutableSet<MethodInvocationSite> record();

  public static InvocationSiteTransformationRecordBuilder builder() {
    return new AutoValue_InvocationSiteTransformationRecord.Builder();
  }

  /** The builder for {@link InvocationSiteTransformationRecord}. */
  @AutoValue.Builder
  public abstract static class InvocationSiteTransformationRecordBuilder {

    abstract ImmutableSet.Builder<MethodInvocationSite> recordBuilder();

    public final InvocationSiteTransformationRecordBuilder addTransformation(
        MethodInvocationSite originalMethodInvocationSite) {
      recordBuilder().add(originalMethodInvocationSite);
      return this;
    }

    public abstract InvocationSiteTransformationRecord build();
  }
}
