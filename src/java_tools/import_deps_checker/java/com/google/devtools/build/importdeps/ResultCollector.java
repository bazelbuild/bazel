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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.importdeps.AbstractClassEntryState.IncompleteState;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;

/** The collector that saves all the missing classes. */
public class ResultCollector {

  private final HashSet<String> missingClasss = new HashSet<>();
  private final HashMap<String, IncompleteState> incompleteClasses = new HashMap<>();
  private final HashSet<MemberInfo> missingMembers = new HashSet<>();

  public ResultCollector() {}

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
      missingClasss.add(state.asIncompleteState().getMissingAncestor()); // Add the real missing.
    } else if (state.isMissingState()) {
      missingClasss.add(internalName);
    } else {
      throw new UnsupportedOperationException("Unsupported state " + state);
    }
  }

  /** Returns {@literal true} if there is NO dependency issue, {@literal false} otherwise. */
  public boolean isEmpty() {
    return missingClasss.isEmpty() && incompleteClasses.isEmpty() && missingMembers.isEmpty();
  }

  public void addMissingMember(MemberInfo member) {
    missingMembers.add(member);
  }

  public ImmutableList<String> getSortedMissingClassInternalNames() {
    return ImmutableList.sortedCopyOf(missingClasss);
  }

  public ImmutableList<IncompleteState> getSortedIncompleteClasses() {
    return ImmutableList.sortedCopyOf(
        Comparator.comparing(a -> a.classInfo().get().internalName()), incompleteClasses.values());
  }

  public ImmutableList<MemberInfo> getSortedMissingMembers() {
    return ImmutableList.sortedCopyOf(missingMembers);
  }
}
