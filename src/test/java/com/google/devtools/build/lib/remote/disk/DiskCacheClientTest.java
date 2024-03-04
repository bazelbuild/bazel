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
package com.google.devtools.build.lib.remote.disk;

import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnCheckingCacheEvent;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DiskCacheClient}. */
@RunWith(JUnit4.class)
public class DiskCacheClientTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final ExecutorService executorService =
      MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(1));

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path root = fs.getPath("/");
  private DiskCacheClient client;
  private RemoteActionExecutionContext context;

  @Before
  public void setUp() throws Exception {
    client = new DiskCacheClient(root, DIGEST_UTIL, executorService, /* verifyDownloads= */ true);
    context =
        RemoteActionExecutionContext.create(
            mock(Spawn.class),
            mock(SpawnExecutionContext.class),
            RequestMetadata.getDefaultInstance());
  }

  @Test
  public void testSpawnCheckingCacheEvent() throws Exception {
    var unused =
        getFromFuture(
            client.downloadActionResult(
                context,
                DIGEST_UTIL.asActionKey(DIGEST_UTIL.computeAsUtf8("key")),
                /* inlineOutErr= */ false));

    verify(context.getSpawnExecutionContext()).report(SpawnCheckingCacheEvent.create("disk-cache"));
  }
}
