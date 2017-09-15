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

import static com.google.common.base.Preconditions.checkState;

import com.android.dex.Dex;
import java.io.Closeable;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Wrapper around a {@link ZipOutputStream} to simplify writing archives with {@code .dex} files.
 * Adding files generally requires a {@link ZipEntry} in order to control timestamps.
 */
// TODO(kmb): Remove this class and inline into DexFileAggregator
class DexFileArchive implements Closeable {

  private final ZipOutputStream out;

  /**
   * Used to ensure writes from different threads are sequenced, which {@link DexFileAggregator}
   * ensures by making the writer futures wait on each oter.
   */
  private final AtomicReference<ZipEntry> inUse = new AtomicReference<>(null);

  public DexFileArchive(ZipOutputStream out) {
    this.out = out;
  }

  /**
   * Adds a {@code .dex} file with the given details.
   */
  public DexFileArchive addFile(ZipEntry entry, Dex dex) throws IOException {
    checkState(inUse.compareAndSet(null, entry), "Already in use");
    entry.setSize(dex.getLength());
    out.putNextEntry(entry);
    dex.writeTo(out);
    out.closeEntry();
    checkState(inUse.compareAndSet(entry, null), "Swooped in: ", inUse.get());
    return this;
  }

  @Override
  public void close() throws IOException {
    checkState(inUse.get() == null, "Still in use: ", inUse.get());
    out.close();
  }
}
