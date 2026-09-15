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

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import javax.annotation.Nullable;

/** The resolution failure path. */
@AutoValue
public abstract class ResolutionFailureChain {

  public static ResolutionFailureChain createMissingClass(String missingClass) {
    return new AutoValue_ResolutionFailureChain(
        ImmutableList.of(missingClass),
        /*resolutionStartClass=*/ null,
        /*parentChains=*/ ImmutableList.of());
  }

  public static ResolutionFailureChain createWithParent(
      ClassInfo resolutionStartClass, ImmutableList<ResolutionFailureChain> parentChains) {
    Preconditions.checkArgument(!parentChains.isEmpty(), "The parentChains cannot be empty.");
    return new AutoValue_ResolutionFailureChain(
        parentChains
            .stream()
            .flatMap(chain -> chain.missingClasses().stream())
            .sorted()
            .distinct()
            .collect(ImmutableList.toImmutableList()),
        resolutionStartClass,
        parentChains);
  }

  /** The missing class that causes the resolution failure. */
  public abstract ImmutableList<String> missingClasses();

  /** The start of this resolution chain. */
  @Nullable
  public abstract ClassInfo resolutionStartClass();

  /** The resolution chain of the parent class. */
  public abstract ImmutableList<ResolutionFailureChain> parentChains();

  /** For all the missing classes, represent the first chains that lead to the missing classes. */
  public ImmutableMultimap<String, ClassInfo> getMissingClassesWithSubclasses() {
    ImmutableMultimap.Builder<String, ClassInfo> result = ImmutableMultimap.builder();
    getMissingClassesWithSubclasses(resolutionStartClass(), this.parentChains(), result);
    return result.build();
  }

  private static void getMissingClassesWithSubclasses(
      ClassInfo subclass,
      ImmutableList<ResolutionFailureChain> parentChains,
      ImmutableMultimap.Builder<String, ClassInfo> result) {
    for (ResolutionFailureChain parentChain : parentChains) {
      if (parentChain.resolutionStartClass() == null) {
        checkState(
            parentChain.parentChains().isEmpty() && parentChain.missingClasses().size() == 1);
        result.put(parentChain.missingClasses().get(0), subclass);
      } else {
        checkState(!parentChain.parentChains().isEmpty());
        getMissingClassesWithSubclasses(
            parentChain.resolutionStartClass(), parentChain.parentChains(), result);
      }
    }
  }
}
