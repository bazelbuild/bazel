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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Preconditions.checkState;

import com.google.protobuf.CodedOutputStream;
import java.util.ArrayList;

/**
 * Records a profile into a given {@link ProfileCollector} for a single serialization thread.
 *
 * <p>The client should call the {@link #pushLocation} when entering serialization of an object then
 * {@link #recordBytesAndPopLocation} when that object's serialization completes. Since
 * serialization is a recursive, this typically means the number of pushes will be greater than the
 * number of pops while serialization is ongoing, but must eventually balance.
 */
final class ProfileRecorder {
  private final ProfileCollector profileCollector;
  private final ArrayList<String> locationStack = new ArrayList<>();

  ProfileRecorder(ProfileCollector profileCollector) {
    this.profileCollector = profileCollector;
  }

  void pushLocation(ObjectCodec<?> codec) {
    locationStack.add(codec.getClass().getCanonicalName());
  }

  void recordBytesAndPopLocation(int startBytes, CodedOutputStream codedOut) {
    int bytesWritten = codedOut.getTotalBytesWritten();
    checkState(bytesWritten >= startBytes);
    profileCollector.recordSample(locationStack, bytesWritten - startBytes);
    locationStack.remove(locationStack.size() - 1);
  }

  ProfileCollector getProfileCollector() {
    return profileCollector;
  }

  void checkStackEmpty(Object subjectForContext) {
    checkState(
        locationStack.isEmpty(), "subject=%s, locationStack=%s", subjectForContext, locationStack);
  }
}
