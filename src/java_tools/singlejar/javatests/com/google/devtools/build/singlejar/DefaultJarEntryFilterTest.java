// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.singlejar;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.Arrays;
import java.util.Date;
import java.util.jar.JarFile;

/**
 * Unit tests for {@link DefaultJarEntryFilter}.
 */
@RunWith(JUnit4.class)
public class DefaultJarEntryFilterTest {

  private static final Date DOS_EPOCH = ZipCombiner.DOS_EPOCH;

  @Test
  public void testSingleInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("abc", callback);
    assertEquals(Arrays.asList("copy"), callback.calls);
    assertEquals(Arrays.asList(DOS_EPOCH), callback.dates);
  }

  @Test
  public void testProtobufExtensionsInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("protobuf.meta", callback);
    assertEquals(Arrays.asList("customMerge"), callback.calls);
    assertEquals(Arrays.asList(DOS_EPOCH), callback.dates);
  }

  @Test
  public void testManifestInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept(JarFile.MANIFEST_NAME, callback);
    assertEquals(Arrays.asList("skip"), callback.calls);
  }

  @Test
  public void testServiceInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("META-INF/services/any.service", callback);
    assertEquals(Arrays.asList("customMerge"), callback.calls);
    assertEquals(Arrays.asList(DOS_EPOCH), callback.dates);
  }

  @Test
  public void testSpringHandlers() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("META-INF/spring.handlers", callback);
    assertEquals(Arrays.asList("customMerge"), callback.calls);
    assertEquals(Arrays.asList(DOS_EPOCH), callback.dates);
  }

  @Test
  public void testSpringSchemas() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("META-INF/spring.schemas", callback);
    assertEquals(Arrays.asList("customMerge"), callback.calls);
    assertEquals(Arrays.asList(DOS_EPOCH), callback.dates);
  }

  @Test
  public void testClassInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("a.class", callback);
    assertEquals(Arrays.asList("copy"), callback.calls);
    assertEquals(Arrays.asList(DefaultJarEntryFilter.DOS_EPOCH_PLUS_2_SECONDS), callback.dates);
  }

  @Test
  public void testOtherSkippedInputs() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    ZipEntryFilter filter = new DefaultJarEntryFilter();
    filter.accept("a.SF", callback);
    filter.accept("a.DSA", callback);
    filter.accept("a.RSA", callback);
    assertEquals(Arrays.asList("skip", "skip", "skip"), callback.calls);
    assertEquals(Arrays.<Date>asList(), callback.dates);
  }
}
