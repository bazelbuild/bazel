// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.importdeps;


import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import java.util.Optional;

/**
 * The state for a class entry used in {@link ClassCache}. A state can be
 *
 * <ul>
 *   <li>EXISTING: this class exists.
 *   <li>INCOMPLETE: this class exists, but at least one of its ancestor is missing.
 *   <li>MISSING: this class is missing on the classpath.
 * </ul>
 */
public abstract class AbstractClassEntryState {

  public boolean isMissingState() {
    return this instanceof MissingState;
  }

  public MissingState asMissingState() {
    throw new IllegalStateException("Not a missing state " + this);
  }

  public boolean isExistingState() {
    return this instanceof ExistingState;
  }

  public ExistingState asExistingState() {
    throw new IllegalStateException("Not an existing state " + this);
  }

  public boolean isIncompleteState() {
    return this instanceof IncompleteState;
  }

  public IncompleteState asIncompleteState() {
    throw new IllegalStateException("Not an incomplete state " + this);
  }

  public abstract Optional<ClassInfo> classInfo();

  /** A state to indicate that a class exists. */
  @AutoValue
  public abstract static class ExistingState extends AbstractClassEntryState {

    public static ExistingState create(ClassInfo classInfo) {
      return new AutoValue_AbstractClassEntryState_ExistingState(Optional.of(classInfo));
    }

    @Override
    public ExistingState asExistingState() {
      return this;
    }
  }

  /** A state to indicate that a class is missing. */
  public static final class MissingState extends AbstractClassEntryState {

    private static final MissingState SINGLETON = new MissingState();

    public static MissingState singleton() {
      return SINGLETON;
    }

    private MissingState() {}

    @Override
    public MissingState asMissingState() {
      return this;
    }

    @Override
    public Optional<ClassInfo> classInfo() {
      return Optional.empty();
    }
  }

  /**
   * A state to indicate that a class is incomplete, that is, some ancesotor is missing on the
   * classpath.
   */
  @AutoValue
  public abstract static class IncompleteState extends AbstractClassEntryState {

    public static IncompleteState create(
        ClassInfo classInfo, ResolutionFailureChain resolutionFailureChain) {
      return new AutoValue_AbstractClassEntryState_IncompleteState(
          Optional.of(classInfo), resolutionFailureChain);
    }

    public abstract ResolutionFailureChain resolutionFailureChain();

    public ImmutableList<String> missingAncestors() {
      return resolutionFailureChain().missingClasses();
    }

    @Override
    public IncompleteState asIncompleteState() {
      return this;
    }
  }
}
