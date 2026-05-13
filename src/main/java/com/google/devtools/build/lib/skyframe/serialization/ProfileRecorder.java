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
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.skyframe.serialization.ProfileCollector.Counts;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.WriteStatus;
import com.google.protobuf.CodedOutputStream;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Records a profile into a given {@link ProfileCollector} for a single serialization thread.
 *
 * <p>The client should call the {@link #pushLocation} when entering serialization of an object then
 * {@link #recordBytesAndPopLocation} when that object's serialization completes. Since
 * serialization is a recursive, this typically means the number of pushes will be greater than the
 * number of pops while serialization is ongoing, but must eventually balance.
 *
 * <p>This recorder buffers samples internally until a {@link WriteStatus} completes. If the write
 * was novel, the samples are merged into the global {@link ProfileCollector}.
 */
public final class ProfileRecorder implements FutureCallback<Boolean> {
  private final ProfileCollector profileCollector;
  private final ArrayList<ProfilerLocationProvider> locationStack = new ArrayList<>();
  private final HashMap<ImmutableList<ProfilerLocationProvider>, Counts> bufferedSamples =
      new HashMap<>();
  private double byteScale = 1.0;

  public ProfileRecorder(ProfileCollector profileCollector) {
    this.profileCollector = profileCollector;
  }

  public void pushLocation(ProfilerLocationProvider provider) {
    locationStack.add(provider);
  }

  /** Records the given {@code byteCount} at the current location. */
  public void recordBytes(int byteCount) {
    ImmutableList<ProfilerLocationProvider> stack =
        profileCollector.getCanonicalStack(locationStack);

    Counts counts = bufferedSamples.computeIfAbsent(stack, Counts::new);
    counts.count().getAndIncrement();
    counts.totalBytes().getAndAdd(byteCount);
  }

  /** Pops the current location from the stack. */
  public void popLocation() {
    locationStack.remove(locationStack.size() - 1);
  }

  public void recordBytesAndPopLocation(int startBytes, CodedOutputStream codedOut) {
    int bytesWritten = codedOut.getTotalBytesWritten();
    checkState(bytesWritten >= startBytes);

    recordBytes(bytesWritten - startBytes);
    popLocation();
  }

  /**
   * Sets a multiplier for all recorded byte counts to account for compression.
   *
   * <p>This should be called if compression is detected and before {@link #registerWriteStatus}.
   */
  public void setByteScale(double byteScale) {
    this.byteScale = byteScale;
  }

  /**
   * Registers a {@link WriteStatus} to trigger the merge of buffered samples.
   *
   * <p>If {@code status} completes with {@code true}, the samples are recorded in the collector.
   */
  public void registerWriteStatus(WriteStatus status) {
    Futures.addCallback(status, this, directExecutor());
  }

  @Override
  public void onSuccess(Boolean wasNovel) {
    if (!wasNovel) {
      return; // Discards the buffered samples.
    }
    if (byteScale != 1.0) {
      // Applies the scaling factor uniformly to all samples.
      for (Counts counts : bufferedSamples.values()) {
        int scaledBytes = (int) Math.round(counts.totalBytes().get() * byteScale);
        counts.totalBytes().set(scaledBytes);
      }
    }
    profileCollector.recordSamples(bufferedSamples);
  }

  @Override
  public void onFailure(Throwable t) {
    // Discard buffered samples on failure.
  }

  ProfileCollector getProfileCollector() {
    return profileCollector;
  }

  void checkStackEmpty(Object subjectForContext) {
    checkState(
        locationStack.isEmpty(), "subject=%s, locationStack=%s", subjectForContext, locationStack);
  }
}
