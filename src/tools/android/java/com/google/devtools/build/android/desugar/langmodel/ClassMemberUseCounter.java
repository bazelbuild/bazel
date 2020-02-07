/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
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

import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

/** The counter used to track a class member use. */
public final class ClassMemberUseCounter implements TypeMappable<ClassMemberUseCounter> {

  /** Tracks a class member with its associated count. */
  private final ConcurrentMap<ClassMemberUse, LongAdder> memberUseCounter;

  public ClassMemberUseCounter(ConcurrentMap<ClassMemberUse, LongAdder> memberUseCounter) {
    this.memberUseCounter = memberUseCounter;
  }

  /** Increases the member use count by one when an member access is encountered. */
  public void incrementMemberUseCount(ClassMemberUse classMemberUse) {
    memberUseCounter.computeIfAbsent(classMemberUse, k -> new LongAdder()).increment();
  }

  /** Retrieves the total use count of a given class member. */
  public long getMemberUseCount(ClassMemberUse memberKey) {
    return memberUseCounter.getOrDefault(memberKey, new LongAdder()).longValue();
  }

  @Override
  public ClassMemberUseCounter acceptTypeMapper(TypeMapper typeMapper) {
    return new ClassMemberUseCounter(
        memberUseCounter.keySet().stream()
            .collect(
                Collectors.toConcurrentMap(
                    memberUse -> memberUse.acceptTypeMapper(typeMapper), memberUseCounter::get)));
  }
}
