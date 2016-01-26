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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
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
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Map;

/** Tests SingleBuildFileCache. */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public class SingleBuildFileCacheTest {
  private static final String EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e";

  private FileSystem fs;
  private Map<String, Integer> calls;
  private Map<String, byte[]> md5Overrides;

  private SingleBuildFileCache underTest;

  @Before
  public final void setUp() throws Exception {
    calls = new HashMap<>();
    md5Overrides = new HashMap<>();
    fs = new InMemoryFileSystem() {
        @Override
        protected InputStream getInputStream(Path path) throws IOException {
          int c = calls.containsKey(path.toString())
              ? calls.get(path.toString()) : 0;
          c++;
          calls.put(path.toString(), c);
          return super.getInputStream(path);
        }

        @Override
        protected byte[] getMD5Digest(Path path) throws IOException {
          byte[] override = md5Overrides.get(path.getPathString());
          return override != null ? override : super.getMD5Digest(path);
        }
      };
    underTest = new SingleBuildFileCache("/", fs);
    Path root = fs.getRootDirectory();
    Path file = root.getChild("empty");
    file.getOutputStream().close();
  }

  @Test
  public void testExceptionsCached() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/noexist");
    IOException caught = null;
    try {
      underTest.getDigest(empty);
      fail("non existent file should raise exception");
    } catch (IOException expected) {
      caught = expected;
    }
    try {
      underTest.getSizeInBytes(empty);
      fail("non existent file should raise exception.");
    } catch (IOException expected) {
      assertSame(caught, expected);
    }
  }

  @Test
  public void testCache() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/empty");
    underTest.getDigest(empty);
    assert(calls.containsKey("/empty"));
    assertEquals(1, (int) calls.get("/empty"));
    underTest.getDigest(empty);
    assertEquals(1, (int) calls.get("/empty"));
  }

  @Test
  public void testBasic() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/empty");
    assertEquals(0, underTest.getSizeInBytes(empty));
    ByteString digest = underTest.getDigest(empty);

    assertEquals(EMPTY_MD5, digest.toStringUtf8());
    assertEquals("/empty", underTest.getInputFromDigest(digest).getExecPathString());
    assert(underTest.contentsAvailableLocally(digest));

    ByteString other = ByteString.copyFrom("f41d8cd98f00b204e9800998ecf8427e", "UTF-16");
    assert(!underTest.contentsAvailableLocally(other));
    assert(calls.containsKey("/empty"));
  }

  @Test
  public void testUnreadableFileWhenFileSystemSupportsDigest() throws Exception {
    byte[] expectedDigestRaw = MessageDigest.getInstance("md5").digest(
        "randomtext".getBytes(StandardCharsets.UTF_8));
    ByteString expectedDigestEncoded = ByteString.copyFromUtf8(
        BaseEncoding.base16().lowerCase().encode(expectedDigestRaw));
    md5Overrides.put("/unreadable", expectedDigestRaw);

    ActionInput input = ActionInputHelper.fromPath("/unreadable");
    Path file = fs.getPath("/unreadable");
    file.getOutputStream().close();
    file.chmod(0);
    ByteString actualDigest = underTest.getDigest(input);
    assertThat(expectedDigestEncoded).isEqualTo(actualDigest);
  }
}
