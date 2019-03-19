// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.blobstore;

import com.google.common.base.Preconditions;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.vfs.Path;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link SimpleBlobStore} implementation combining two blob stores. A local disk blob store and a
 * remote http blob store. If a blob isn't found in the first store, the second store is used, and
 * the blob added to the first. Put puts the blob on both stores.
 */
public final class CombinedDiskHttpBlobStore extends OnDiskBlobStore {
  private static final Logger logger = Logger.getLogger(CombinedDiskHttpBlobStore.class.getName());
  private final SimpleBlobStore bsHttp;

  public CombinedDiskHttpBlobStore(Path root, SimpleBlobStore bsHttp) {
    super(root);
    this.bsHttp = Preconditions.checkNotNull(bsHttp);
  }

  @Override
  public boolean containsKey(String key) {
    // HTTP cache does not support containsKey.
    // Don't support it here either for predictable semantics.
    throw new UnsupportedOperationException("HTTP Caching does not use this method.");
  }

  @Override
  public ListenableFuture<Boolean> get(String key, OutputStream out) {
    boolean foundOnDisk = super.containsKey(key);

    if (foundOnDisk) {
      return super.get(key, out);
    } else {
      // Write a temporary file first, and then rename, to avoid data corruption in case of a crash.
      Path temp = toPath(UUID.randomUUID().toString());

      OutputStream tempOut;
      try {
        tempOut = temp.getOutputStream();
      } catch (IOException e) {
        return Futures.immediateFailedFuture(e);
      }
      ListenableFuture<Boolean> chained =
          Futures.transformAsync(
              bsHttp.get(key, tempOut),
              (found) -> {
                if (!found) {
                  return Futures.immediateFuture(false);
                } else {
                  Path target = toPath(key);
                  // The following note and line is taken from OnDiskBlobStore.java
                  // TODO(ulfjack): Fsync temp here before we rename it to avoid data loss in the
                  // case of machine
                  // crashes (the OS may reorder the writes and the rename).
                  temp.renameTo(target);

                  SettableFuture<Boolean> f = SettableFuture.create();
                  try (InputStream in = target.getInputStream()) {
                    ByteStreams.copy(in, out);
                    f.set(true);
                  } catch (IOException e) {
                    f.setException(e);
                  }
                  return f;
                }
              },
              MoreExecutors.directExecutor());
      chained.addListener(
          () -> {
            try {
              tempOut.close();
            } catch (IOException e) {
              // not sure what to do here, we either are here because of another exception being
              // thrown,
              // or we have successfully used the file we are trying (and failing) to close
              logger.log(Level.WARNING, "Failed to close temporary file on get", e);
            }
          },
          MoreExecutors.directExecutor());
      return chained;
    }
  }

  @Override
  public boolean getActionResult(String key, OutputStream out)
      throws IOException, InterruptedException {
    if (super.getActionResult(key, out)) {
      return true;
    }

    try (ByteArrayOutputStream tmpOut = new ByteArrayOutputStream()) {
      if (bsHttp.getActionResult(key, tmpOut)) {
        byte[] tmp = tmpOut.toByteArray();
        super.putActionResult(key, tmp);
        ByteStreams.copy(new ByteArrayInputStream(tmp), out);
        return true;
      }
    }

    return false;
  }

  @Override
  public void put(String key, long length, InputStream in)
      throws IOException, InterruptedException {
    super.put(key, length, in);
    try (InputStream inFile = toPath(key).getInputStream()) {
      bsHttp.put(key, length, inFile);
    }
  }

  @Override
  public void putActionResult(String key, byte[] in) throws IOException, InterruptedException {
    super.putActionResult(key, in);
    bsHttp.putActionResult(key, in);
  }

  @Override
  public void close() {
    super.close();
    bsHttp.close();
  }
}
