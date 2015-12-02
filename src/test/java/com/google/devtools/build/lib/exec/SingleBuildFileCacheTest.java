// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.protobuf.ByteString;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/** Tests SingleBuildFileCache. */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public class SingleBuildFileCacheTest {
  private FileSystem fs;
  private ActionInputFileCache cache;
  private Map<String, Integer> calls;
  private static final String EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e";

  @Before
  public final void setUp() throws Exception {
    calls = new HashMap<>();
    fs = new InMemoryFileSystem() {
        @Override
        protected InputStream getInputStream(Path path) throws IOException {
          int c = calls.containsKey(path.toString())
              ? calls.get(path.toString()) : 0;
          c++;
          calls.put(path.toString(), c);
          return super.getInputStream(path);
        }
      };
    cache = new SingleBuildFileCache("/", fs);
    Path root = fs.getRootDirectory();
    Path file = root.getChild("empty");
    file.getOutputStream().close();
  }

  @Test
  public void testExceptionsCached() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/noexist");
    IOException caught = null;
    try {
      cache.getDigest(empty);
      fail("non existent file should raise exception");
    } catch (IOException expected) {
      caught = expected;
    }
    try {
      cache.getSizeInBytes(empty);
      fail("non existent file should raise exception.");
    } catch (IOException expected) {
      assertSame(caught, expected);
    }
  }

  @Test
  public void testCache() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/empty");
    cache.getDigest(empty);
    assert(calls.containsKey("/empty"));
    assertEquals(1, (int) calls.get("/empty"));
    cache.getDigest(empty);
    assertEquals(1, (int) calls.get("/empty"));
  }

  @Test
  public void testBasic() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/empty");
    assertEquals(0, cache.getSizeInBytes(empty));
    ByteString digest = cache.getDigest(empty);

    assertEquals(EMPTY_MD5, digest.toStringUtf8());
    assertEquals("/empty", cache.getInputFromDigest(digest).getExecPathString());
    assert(cache.contentsAvailableLocally(digest));

    ByteString other = ByteString.copyFrom("f41d8cd98f00b204e9800998ecf8427e", "UTF-16");
    assert(!cache.contentsAvailableLocally(other));
    assert(calls.containsKey("/empty"));
  }
}
