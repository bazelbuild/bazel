// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AsynchronousTreeDeleter}. */
@RunWith(JUnit4.class)
public final class AsynchronousTreeDeleterTest {

  @Test
  public void deleteTree_queueFull_deletesOnCaller() throws Exception {
    CountDownLatch firstDeletionStarted = new CountDownLatch(1);
    CountDownLatch releaseFirstDeletion = new CountDownLatch(1);
    FileSystem fs =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public void deleteTree(PathFragment path) throws IOException {
            if (path.getBaseName().equals("0")) {
              firstDeletionStarted.countDown();
              try {
                releaseFirstDeletion.await();
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException(e);
              }
            }
            super.deleteTree(path);
          }
        };
    Scratch scratch = new Scratch(fs);
    Path trashBase = scratch.dir("/trash");
    AsynchronousTreeDeleter deleter =
        new AsynchronousTreeDeleter(trashBase, /* maxPendingDeletions= */ 1);

    // The only worker blocks on trash tree 0, and trash tree 1 fills the queue.
    deleter.deleteTree(scratch.file("/sandbox/0/input", "0").getParentDirectory());
    firstDeletionStarted.await();
    deleter.deleteTree(scratch.file("/sandbox/1/input", "1").getParentDirectory());

    deleter.deleteTree(scratch.file("/sandbox/2/input", "2").getParentDirectory());

    assertThat(trashBase.getChild("2").exists()).isFalse();
    assertThat(trashBase.getChild("1").exists()).isTrue();

    releaseFirstDeletion.countDown();
    deleter.shutdown();
    assertThat(trashBase.getDirectoryEntries()).isEmpty();
  }
}
