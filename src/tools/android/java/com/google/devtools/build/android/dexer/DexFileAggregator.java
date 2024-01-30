// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.android.dex.Dex;
import com.android.dx.command.dexer.DxContext;
import com.android.dx.merge.CollisionPolicy;
import com.android.dx.merge.DexMerger;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.Closeable;
import java.io.IOException;
import java.nio.BufferOverflowException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.zip.ZipEntry;

/**
 * Merger for {@code .dex} files into larger chunks subject to {@code .dex} file limits on methods
 * and fields.
 */
class DexFileAggregator implements Closeable {

  /**
   * File extension of a {@code .dex} file.
   */
  private static final String DEX_EXTENSION = ".dex";

  private final ArrayList<Dex> currentShard = new ArrayList<>();
  private final boolean forceJumbo;
  private final int wasteThresholdPerDex;
  private final MultidexStrategy multidex;
  private final DxContext context;
  private final ListeningExecutorService executor;
  private final DexFileArchive dest;
  private final String dexPrefix;
  private final DexLimitTracker tracker;

  private int nextDexFileIndex = 0;
  private ListenableFuture<Void> lastWriter = Futures.<Void>immediateFuture(null);

  public DexFileAggregator(
      DxContext context,
      DexFileArchive dest,
      ListeningExecutorService executor,
      MultidexStrategy multidex,
      boolean forceJumbo,
      int maxNumberOfIdxPerDex,
      int wasteThresholdPerDex,
      String dexPrefix) {
    this.context = context;
    this.dest = dest;
    this.executor = executor;
    this.multidex = multidex;
    this.forceJumbo = forceJumbo;
    this.wasteThresholdPerDex = wasteThresholdPerDex;
    this.dexPrefix = dexPrefix;
    tracker = new DexLimitTracker(maxNumberOfIdxPerDex);
  }

  @CanIgnoreReturnValue
  public DexFileAggregator add(Dex dexFile) {
    if (multidex.isMultidexAllowed()) {
      // To determine whether currentShard is "full" we track unique field and method signatures,
      // which predicts precisely the number of field and method indices.
      tracker.track(dexFile);
      if (tracker.outsideLimits() && !currentShard.isEmpty()) {
        // For simplicity just start a new shard to fit the given file.
        // Don't bother with waiting for a later file that might fit the old shard as in the extreme
        // we'd have to wait until the end to write all shards.
        rotateDexFile();
        tracker.track(dexFile);
      }
    }
    currentShard.add(dexFile);
    return this;
  }

  @Override
  public void close() throws IOException {
    try {
      if (!currentShard.isEmpty()) {
        rotateDexFile();
      }
      // Wait for last shard to be written before closing underlying archive
      lastWriter.get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    } catch (ExecutionException e) {
      Throwables.throwIfInstanceOf(e.getCause(), IOException.class);
      Throwables.throwIfUnchecked(e.getCause());
      throw new AssertionError("Unexpected execution exception", e);
    } finally {
      dest.close();
    }
  }

  public void flush() {
    checkState(multidex.isMultidexAllowed());
    if (!currentShard.isEmpty()) {
      rotateDexFile();
    }
  }

  public int getDexFilesWritten() {
    return nextDexFileIndex;
  }

  private void rotateDexFile() {
    writeMergedFile(currentShard.toArray(/* apparently faster than pre-sized array */ new Dex[0]));
    currentShard.clear();
    tracker.clear();
  }

  private void writeMergedFile(Dex... dexes) {
    checkArgument(0 < dexes.length);
    checkState(multidex.isMultidexAllowed() || nextDexFileIndex == 0);
    String filename = getDexFileName(nextDexFileIndex++);
    ListenableFuture<Dex> merged =
        dexes.length == 1 && !forceJumbo
            ? Futures.immediateFuture(dexes[0])
            : executor.submit(new RunDexMerger(dexes));
    lastWriter =
        Futures.whenAllSucceed(lastWriter, merged)
            .call(new WriteFile(filename, merged, dest), executor);
  }

  private Dex merge(Dex... dexes) throws IOException {
    switch (dexes.length) {
      case 0:
        return new Dex(0);
      case 1:
        // Need to actually process the single given file for forceJumbo :(
        return forceJumbo ? merge(dexes[0], new Dex(0)) : dexes[0];
      default: // fall out
    }
    DexMerger dexMerger = new DexMerger(dexes, CollisionPolicy.FAIL, context);
    dexMerger.setCompactWasteThreshold(wasteThresholdPerDex);
    if (forceJumbo) {
      try {
        DexMerger.class.getMethod("setForceJumbo", Boolean.TYPE).invoke(dexMerger, true);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException("--forceJumbo flag not supported", e);
      }
    }

    try {
      return dexMerger.merge();
    } catch (BufferOverflowException e) {
      if (dexes.length <= 2) {
        throw e;
      }
      // Bug in dx can cause this for ~1500 or more classes
      Dex[] left = Arrays.copyOf(dexes, dexes.length / 2);
      Dex[] right = Arrays.copyOfRange(dexes, left.length, dexes.length);
      System.err.printf("Couldn't merge %d classes, trying %d%n", dexes.length, left.length);
      try {
        return merge(merge(left), merge(right));
      } catch (RuntimeException e2) {
        e2.addSuppressed(e);
        throw e2;
      }
    }
  }

  // More or less copied from from com.android.dx.command.dexer.Main
  private String getDexFileName(int i) {
    return dexPrefix + (i == 0 ? "" : i + 1) + DEX_EXTENSION;
  }


  private class RunDexMerger implements Callable<Dex> {

    private final Dex[] dexes;

    public RunDexMerger(Dex... dexes) {
      this.dexes = dexes;
    }

    @Override
    public Dex call() throws IOException {
      try {
        return merge(dexes);
      } catch (Throwable t) {
        // Print out exceptions so they don't get swallowed completely
        t.printStackTrace();
        Throwables.throwIfInstanceOf(t, IOException.class);
        Throwables.throwIfUnchecked(t);
        throw new AssertionError(t);  // shouldn't get here
      }
    }
  }

  private static class WriteFile implements Callable<Void> {

    private final ListenableFuture<Dex> dex;
    private final String filename;
    @SuppressWarnings ("hiding") private final DexFileArchive dest;

    public WriteFile(String filename, ListenableFuture<Dex> dex, DexFileArchive dest) {
      this.filename = filename;
      this.dex = dex;
      this.dest = dest;
    }

    @Override
    public Void call() throws Exception {
      try {
        checkState(dex.isDone());
        ZipEntry entry = new ZipEntry(filename);
        entry.setTime(0L); // Use simple stable timestamps for deterministic output
        dest.addFile(entry, dex.get());
        return null;
      } catch (Exception e) {
        // Print out exceptions so they don't get swallowed completely
        e.printStackTrace();
        throw e;
      } catch (Throwable t) {
        t.printStackTrace();
        Throwables.throwIfUnchecked(t);
        throw new AssertionError(t);  // shouldn't get here
      }
    }
  }
}
