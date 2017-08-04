// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.MoreExecutors.newDirectExecutorService;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.when;

import com.android.dex.Dex;
import com.android.dx.command.dexer.DxContext;
import com.android.dx.dex.DexOptions;
import com.android.dx.dex.cf.CfOptions;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.dexer.Dexing.DexingKey;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.Enumeration;
import java.util.concurrent.Future;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link DexConversionEnqueuer}. */
@RunWith(JUnit4.class)
public class DexConversionEnqueuerTest {

  private static final long FILE_TIME = 12345678987654321L;

  @Mock private ZipFile zip;

  private DexConversionEnqueuer stuffer;
  private final Cache<DexingKey, byte[]> cache = CacheBuilder.newBuilder().build();

  @Before
  public void setUp() {
    MockitoAnnotations.initMocks(this);
    makeStuffer();
  }

  private void makeStuffer() {
    stuffer =
        new DexConversionEnqueuer(
            zip,
            newDirectExecutorService(),
            new DexConverter(new Dexing(new DxContext(), new DexOptions(), new CfOptions())),
            cache);
  }

  /** Makes sure there's always a future returning {@code null} at the end. */
  @After
  public void assertEndOfStreamMarker() throws Exception {
    Future<ZipEntryContent> f = stuffer.getFiles().remove();
    assertThat(f.isDone()).isTrue();
    assertThat(f.get()).isNull();
    assertThat(stuffer.getFiles()).isEmpty();
  }

  @Test
  public void testEmptyZip() throws Exception {
    mockEntries();
    stuffer.call();
  }

  @Test
  public void testDirectory_copyEmptyBuffer() throws Exception {
    ZipEntry entry = newZipEntry("dir/", 0);
    assertThat(entry.isDirectory()).isTrue(); // test sanity
    mockEntries(entry);

    stuffer.call();
    Future<ZipEntryContent> f = stuffer.getFiles().remove();
    assertThat(f.isDone()).isTrue();
    assertThat(f.get().getEntry()).isEqualTo(entry);
    assertThat(f.get().getContent()).isEmpty();
    assertThat(entry.getCompressedSize()).isEqualTo(0);
  }

  @Test
  public void testFile_copyContent() throws Exception {
    byte[] content = "Hello".getBytes(UTF_8);
    ZipEntry entry = newZipEntry("file", content.length);
    mockEntries(entry);
    when(zip.getInputStream(entry)).thenReturn(new ByteArrayInputStream(content));

    stuffer.call();
    Future<ZipEntryContent> f = stuffer.getFiles().remove();
    assertThat(f.isDone()).isTrue();
    assertThat(f.get().getEntry()).isEqualTo(entry);
    assertThat(f.get().getContent()).isEqualTo(content);
    assertThat(cache.size()).isEqualTo(0); // don't cache resource files
    assertThat(entry.getCompressedSize()).isEqualTo(-1); // we don't know how the file will compress
  }

  @Test
  public void testClass_convertToDex() throws Exception {
    testConvertClassToDex();
  }

  @Test
  public void testClass_cachedResult() throws Exception {
    byte[] dexcode = testConvertClassToDex();

    makeStuffer();
    String filename = getClass().getName().replace('.', '/') + ".class";
    mockClassFile(filename);
    stuffer.call();
    Future<ZipEntryContent> f = stuffer.getFiles().remove();
    assertThat(f.isDone()).isTrue();
    assertThat(f.get().getEntry().getName()).isEqualTo(filename + ".dex");
    assertThat(f.get().getEntry().getTime()).isEqualTo(FILE_TIME);
    assertThat(f.get().getContent()).isSameAs(dexcode);
  }

  private byte[] testConvertClassToDex() throws Exception {
    String filename = getClass().getName().replace('.', '/') + ".class";
    byte[] bytecode = mockClassFile(filename);

    stuffer.call();
    Future<ZipEntryContent> f = stuffer.getFiles().remove();
    assertThat(f.isDone()).isTrue();
    assertThat(f.get().getEntry().getName()).isEqualTo(filename + ".dex");
    assertThat(f.get().getEntry().getTime()).isEqualTo(FILE_TIME);
    assertThat(f.get().getEntry().getSize()).isEqualTo(-1);
    assertThat(f.get().getEntry().getCompressedSize()).isEqualTo(-1);
    byte[] dexcode = f.get().getContent();
    Dex dex = new Dex(dexcode);
    assertThat(dex.classDefs()).hasSize(1);
    assertThat(cache.getIfPresent(DexingKey.create(false, false, bytecode))).isSameAs(dexcode);
    assertThat(cache.getIfPresent(DexingKey.create(true, false, bytecode))).isNull();
    assertThat(cache.getIfPresent(DexingKey.create(false, true, bytecode))).isNull();
    assertThat(cache.getIfPresent(DexingKey.create(true, true, bytecode))).isNull();
    return dexcode;
  }

  private byte[] mockClassFile(String filename) throws IOException {
    byte[] bytecode = ByteStreams.toByteArray(
        Thread.currentThread().getContextClassLoader().getResourceAsStream(filename));
    ZipEntry entry = newZipEntry(filename, bytecode.length);
    assertThat(entry.isDirectory()).isFalse(); // test sanity
    mockEntries(entry);
    when(zip.getInputStream(entry)).thenReturn(new ByteArrayInputStream(bytecode));
    return bytecode;
  }

  @Test
  public void testException_stillEnqueueEndOfStreamMarker() throws Exception {
    when(zip.entries()).thenThrow(new IllegalStateException("test"));
    try {
      stuffer.call();
      fail("IllegalStateException expected");
    } catch (IllegalStateException expected) {
    }
    // assertEndOfStreamMarker() makes sure the end-of-stream marker is there
  }

  private ZipEntry newZipEntry(String name, long size) {
    ZipEntry result = new ZipEntry(name);
    // Class under test needs sizing information so we need to set it for the test.  These values
    // are always set when reading zip entries from an existing zip file.
    result.setSize(size);
    result.setCompressedSize(size);
    result.setTime(FILE_TIME);
    return result;
  }

  // thenReturn expects a generic type that uses the unknown ? extends ZipEntry "returned"
  // by entries(). Since we can't come up with an unknown type, use raw type to make this typecheck.
  // Note this is safe: actual entries() callers expect ZipEntries and they get them.
  @SuppressWarnings({"rawtypes", "unchecked"})
  private void mockEntries(ZipEntry... entries) {
    when(zip.entries())
        .thenReturn((Enumeration) Collections.enumeration(ImmutableList.copyOf(entries)));
  }
}
