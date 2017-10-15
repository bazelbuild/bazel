// Copyright 2017 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec.apple;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link CacheManager}.
 */
@RunWith(JUnit4.class)
public class CacheManagerTest {
  private static final String CACHE_FILENAME = "cachefilename";
  private Path cachePath;
  private CacheManager cacheManager;
  private final InMemoryFileSystem fs = new InMemoryFileSystem();

  @Before
  public void init() throws Exception {
    Path outputBase = fs.getPath("/somewhere");
    assertThat(outputBase.createDirectory()).isTrue();
    cachePath = outputBase.getRelative(CACHE_FILENAME);
    cacheManager = new CacheManager(outputBase, CACHE_FILENAME);
  }
  
  @Test
  public void testEmptyCache() throws Exception {
    assertThat(cacheManager.getValue("foo")).isNull();
  }
  
  @Test
  public void testSingleKeyCacheHits() throws Exception {
    cacheManager.writeEntry(ImmutableList.of("foo"), "sdkroot1");
    cacheManager.writeEntry(ImmutableList.of("bar"), "sdkroot2");
    cacheManager.writeEntry(ImmutableList.of("baz"), "sdkroot3");
    
    assertThat(cacheManager.getValue("bar")).isEqualTo("sdkroot2");
    assertThat(cacheManager.getValue("baz")).isEqualTo("sdkroot3");
    assertThat(cacheManager.getValue("foo")).isEqualTo("sdkroot1");
  }
  
  @Test
  public void testSingleKeyCacheMiss() throws Exception {
    cacheManager.writeEntry(ImmutableList.of("foo"), "sdkroot1");

    assertThat(cacheManager.getValue("bar")).isNull();
  }
  
  @Test
  public void testMultiKeyCacheHits() throws Exception {
    cacheManager.writeEntry(ImmutableList.of("foo", "cat"), "sdkroot1");
    cacheManager.writeEntry(ImmutableList.of("bar", "cat"), "sdkroot2");
    cacheManager.writeEntry(ImmutableList.of("baz", "dog"), "sdkroot3");
    
    assertThat(cacheManager.getValue("bar", "cat")).isEqualTo("sdkroot2");
    assertThat(cacheManager.getValue("baz", "dog")).isEqualTo("sdkroot3");
    assertThat(cacheManager.getValue("foo", "cat")).isEqualTo("sdkroot1");
  }
  
  @Test
  public void testMultiKeyCacheMiss() throws Exception {
    cacheManager.writeEntry(ImmutableList.of("foo", "cat"), "sdkroot1");

    assertThat(cacheManager.getValue("bar", "cat")).isNull();
  }
  
  @Test
  public void testKeyCountMismatch() throws Exception {
    cacheManager.writeEntry(ImmutableList.of("foo", "cat"), "sdkroot1");

    try {
      cacheManager.getValue("foo");
      fail("Key count mismatch, should have thrown exception");
    } catch (IllegalStateException expected) {
      assertThat(expected).hasMessageThat().contains("malformed");
    }
  }

  @Test
  public void testBadCache() throws Exception {
    FileSystemUtils.writeContentAsLatin1(cachePath, "blah blah blah");

    try {
      cacheManager.getValue("foo");
      fail("Cache file was corrupt, should have thrown exception");
    } catch (IllegalStateException expected) {
      assertThat(expected).hasMessageThat().contains("malformed");
    }
  }
}
