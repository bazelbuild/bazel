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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.importdeps.AbstractClassEntryState.IncompleteState;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;

/** The collector that saves all the missing classes. */
public class ResultCollector {

  private final HashSet<String> missingClasss = new HashSet<>();
  private final HashMap<String, IncompleteState> incompleteClasses = new HashMap<>();
  private final HashSet<MissingMember> missingMembers = new HashSet<>();
  private final HashSet<Path> indirectDeps = new HashSet<>();
  private final boolean checkMissingMembers;

  public ResultCollector(boolean checkMissingMembers) {
    this.checkMissingMembers = checkMissingMembers;
  }

  public void addMissingOrIncompleteClass(String internalName, AbstractClassEntryState state) {
    checkArgument(
        internalName.length() > 0 && Character.isJavaIdentifierStart(internalName.charAt(0)),
        "The internal name is invalid. %s",
        internalName);
    if (state.isIncompleteState()) {
      IncompleteState oldValue = incompleteClasses.put(internalName, state.asIncompleteState());
      checkState(
          oldValue == null || oldValue == state,
          "The old value and the new value are not the same object. old=%s, new=%s",
          oldValue,
          state);
      // Add the real missing.
      state.asIncompleteState().missingAncestors().forEach(missingClasss::add);
    } else if (state.isMissingState()) {
      missingClasss.add(internalName);
    } else {
      throw new UnsupportedOperationException("Unsupported state " + state);
    }
  }

  /** Returns {@literal true} if there is NO dependency issue, {@literal false} otherwise. */
  public boolean isEmpty() {
    return missingClasss.isEmpty()
        && incompleteClasses.isEmpty()
        && missingMembers.isEmpty()
        && indirectDeps.isEmpty();
  }

  /** Returns {@code true} if we want to report missing members, {@code false} otherwise. */
  public boolean getCheckMissingMembers() {
    return checkMissingMembers;
  }

  public void addMissingMember(ClassInfo owner, MemberInfo member) {
    if (checkMissingMembers) {
      missingMembers.add(MissingMember.create(owner, member));
    }
  }

  public void addIndirectDep(Path indirectDep) {
    indirectDeps.add(indirectDep);
  }

  public ImmutableList<String> getSortedMissingClassInternalNames() {
    return ImmutableList.sortedCopyOf(missingClasss);
  }

  public ImmutableList<IncompleteState> getSortedIncompleteClasses() {
    return ImmutableList.sortedCopyOf(
        Comparator.comparing(a -> a.classInfo().get().internalName()), incompleteClasses.values());
  }

  public ImmutableList<MissingMember> getSortedMissingMembers() {
    return ImmutableList.sortedCopyOf(missingMembers);
  }

  public ImmutableList<Path> getSortedIndirectDeps() {
    return ImmutableList.sortedCopyOf(indirectDeps);
  }

  /**
   * A missing member on the classpath. This class does not contain the member name and description,
   * but also the owner of the member.
   */
  @AutoValue
  public abstract static class MissingMember implements Comparable<MissingMember> {

    public static MissingMember create(ClassInfo owner, String memberName, String descriptor) {
      return create(owner, MemberInfo.create(memberName, descriptor));
    }

    public static MissingMember create(ClassInfo owner, MemberInfo member) {
      return new AutoValue_ResultCollector_MissingMember(owner, member);
    }

    public abstract ClassInfo owner();

    public abstract MemberInfo member();

    public final String memberName() {
      return member().memberName();
    }

    public final String descriptor() {
      return member().descriptor();
    }

    @Override
    public int compareTo(MissingMember other) {
      return ComparisonChain.start()
          .compare(this.owner(), other.owner())
          .compare(this.memberName(), other.memberName())
          .compare(this.descriptor(), other.descriptor())
          .result();
    }
  }
}
