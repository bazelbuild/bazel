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
package com.google.devtools.build.lib.bazel.repository.cache;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.spy;

import com.google.common.collect.Collections2;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.FileSystemLock;
import com.google.devtools.build.lib.util.FileSystemLock.LockMode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.util.Collection;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LocalRepoContentsCache}. */
@RunWith(JUnit4.class)
public final class LocalRepoContentsCacheTest {
  @Test
  public void gc_releasesLockBetweenEntries() throws Exception {
    var root = TestUtils.createUniqueTmpDir(null).asFragment();
    var sharedLocksAcquired = new AtomicInteger();
    var fs =
        new JavaIoFileSystem(DigestHashFunction.SHA256) {
          @Override
          public Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
              throws IOException {
            var entries = super.readdir(path, followSymlinks);
            if (!path.equals(root)) {
              return entries;
            }
            return Collections2.transform(
                entries,
                entry -> {
                  if (entry.getType() == Dirent.Type.DIRECTORY
                      && !entry.getName().equals(LocalRepoContentsCache.TRASH_PATH)) {
                    try (var lock =
                        FileSystemLock.tryGet(
                            getPath(root).getRelative(LocalRepoContentsCache.LOCK_PATH),
                            LockMode.SHARED)) {
                      sharedLocksAcquired.incrementAndGet();
                    } catch (IOException e) {
                      throw new UncheckedIOException(e);
                    }
                  }
                  return entry;
                });
          }
        };
    var cachePath = fs.getPath(root);
    for (String hash : new String[] {"first", "second"}) {
      var entry = cachePath.getRelative(hash);
      entry.getRelative("repo").createDirectoryAndParents();
      var inputs = entry.getRelative("repo.recorded_inputs");
      FileSystemUtils.writeContentAsLatin1(inputs, hash);
      inputs.setLastModifiedTimeNanos(1);
    }
    var cache = new LocalRepoContentsCache();
    cache.setPath(cachePath);

    cache.createGcIdleTask(Duration.ofMillis(1), Duration.ZERO).run();

    assertThat(sharedLocksAcquired.get()).isEqualTo(2);
    for (String hash : new String[] {"first", "second"}) {
      assertThat(cachePath.getRelative(hash).getDirectoryEntries()).isEmpty();
    }
    assertThat(cachePath.getRelative(LocalRepoContentsCache.TRASH_PATH).getDirectoryEntries())
        .isEmpty();
  }

  @Test
  public void gc_preservesMarkerCreatedAfterUnlocking() throws Exception {
    var fileSystem = spy(new JavaIoFileSystem(DigestHashFunction.SHA256));
    Path cacheDir = TestUtils.createUniqueTmpDir(fileSystem);
    var cache = new LocalRepoContentsCache();
    cache.setPath(cacheDir);
    Path trashDir = cacheDir.getChild(LocalRepoContentsCache.TRASH_PATH);
    Path oldRepo = trashDir.getChild(UUID.randomUUID().toString());
    oldRepo.createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(oldRepo.getChild("file"), "old repo contents");
    Path abandonedMarker = trashDir.getChild(UUID.randomUUID().toString());
    FileSystemUtils.writeContentAsLatin1(abandonedMarker, "abandoned marker");
    Path liveMarker = trashDir.getChild(UUID.randomUUID().toString());
    var markerCreated = new AtomicBoolean();

    doAnswer(
            invocation -> {
              PathFragment dir = invocation.getArgument(0);
              if (dir.startsWith(trashDir.asFragment())
                  && markerCreated.compareAndSet(false, true)) {
                // Simulate another server staging a marker in moveToCache as deletion starts.
                // Acquiring the shared lock also checks that GC has released the exclusive lock.
                try (var lock =
                    FileSystemLock.tryGet(
                        cacheDir.getChild(LocalRepoContentsCache.LOCK_PATH), LockMode.SHARED)) {
                  FileSystemUtils.writeContentAsLatin1(liveMarker, "live marker");
                  return invocation.callRealMethod();
                }
              }
              return invocation.callRealMethod();
            })
        .when(fileSystem)
        .deleteTreesBelow(any());

    cache.createGcIdleTask(Duration.ZERO, Duration.ZERO).run();

    assertThat(markerCreated.get()).isTrue();
    assertThat(oldRepo.exists()).isFalse();
    assertThat(abandonedMarker.exists()).isFalse();
    assertThat(new String(FileSystemUtils.readContentAsLatin1(liveMarker)))
        .isEqualTo("live marker");
  }
}
