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

package com.google.devtools.build.lib.bazel.repository.downloader;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache.KeyType;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DownloadManager}. */
@RunWith(JUnit4.class)
public final class DownloadManagerTest {
  @Test
  public void concurrentCacheMissesUseOneDownload() throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var initialLookups = ConcurrentHashMap.<Path>newKeySet();
    var bothMissed = new CountDownLatch(2);
    var cache =
        new DownloadCache() {
          @Override
          public Path get(
              String key, Path destination, KeyType type, String canonicalId, boolean mayHardlink)
              throws IOException, InterruptedException {
            if (initialLookups.add(destination)) {
              // Make both calls see an initial miss before either can fill the cache.
              bothMissed.countDown();
              assertTrue(bothMissed.await(10, SECONDS));
              return null;
            }
            return super.get(key, destination, type, canonicalId, mayHardlink);
          }
        };
    cache.setPath(fs.getPath("/cache"));
    byte[] contents = "pinned content".getBytes(UTF_8);
    var checksum =
        Checksum.fromString(
            KeyType.SHA256, KeyType.SHA256.newHasher().putBytes(contents).hash().toString());
    var downloads = new AtomicInteger();
    Downloader downloader =
        (urls, headers, credentials, digest, canonicalId, output, events, env, type, context) -> {
          downloads.incrementAndGet();
          FileSystemUtils.writeContent(output, contents);
        };
    var manager = new DownloadManager(cache, downloader, null, ExtendedEventHandler.NOOP);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var outputs = new ArrayList<Future<Path>>();
      for (String name : List.of("first", "second")) {
        outputs.add(
            manager.startDownload(
                executor,
                List.of(URI.create("https://example.invalid/" + name)),
                Map.of(),
                Map.of(),
                Optional.of(checksum),
                "archive",
                Optional.empty(),
                fs.getPath("/" + name),
                Map.of(),
                "test download",
                /* mayHardlink= */ false));
      }
      for (var output : outputs) {
        assertArrayEquals(contents, FileSystemUtils.readContent(manager.finalizeDownload(output)));
      }
    }
    assertEquals(1, downloads.get());
  }
}
