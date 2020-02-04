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
import java.util.Optional;

/**
 * A class to hold class-level attribute information.
 *
 * <p>https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-4.html#jvms-4.7
 */
@AutoValue
public abstract class ClassAttributes {

  public abstract String classBinaryName();

  public abstract Optional<String> nestHost();

  public abstract ImmutableSet<String> nestMembers();

  // Include other class attributes as necessary.

  public static ClassAttributesBuilder builder() {
    return new AutoValue_ClassAttributes.Builder();
  }

  /** The builder of {@link ClassAttributes}. */
  @AutoValue.Builder
  public abstract static class ClassAttributesBuilder {

    public abstract ClassAttributesBuilder setClassBinaryName(String classBinaryName);

    public abstract ClassAttributesBuilder setNestHost(String nestHost);

    abstract ImmutableSet.Builder<String> nestMembersBuilder();

    public ClassAttributesBuilder addNestMember(String nestMember) {
      nestMembersBuilder().add(nestMember);
      return this;
    }

    public abstract ClassAttributes build();
  }
}
