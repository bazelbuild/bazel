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

package com.google.devtools.build.singlejar;

import static com.google.common.truth.Truth.assertThat;

import java.io.IOException;
import java.util.Arrays;
import java.util.Date;
import java.util.jar.JarFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
    assertThat(callback.calls).isEqualTo(Arrays.asList("copy"));
    assertThat(callback.dates).isEqualTo(Arrays.asList(DOS_EPOCH));
  }

  @Test
  public void testProtobufExtensionsInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("protobuf.meta", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("customMerge"));
    assertThat(callback.dates).isEqualTo(Arrays.asList(DOS_EPOCH));
  }

  @Test
  public void testManifestInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept(JarFile.MANIFEST_NAME, callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("skip"));
  }

  @Test
  public void testServiceInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("META-INF/services/any.service", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("customMerge"));
    assertThat(callback.dates).isEqualTo(Arrays.asList(DOS_EPOCH));
  }

  @Test
  public void testSpringHandlers() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("META-INF/spring.handlers", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("customMerge"));
    assertThat(callback.dates).isEqualTo(Arrays.asList(DOS_EPOCH));
  }

  @Test
  public void testSpringSchemas() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("META-INF/spring.schemas", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("customMerge"));
    assertThat(callback.dates).isEqualTo(Arrays.asList(DOS_EPOCH));
  }

  @Test
  public void testReferenceConfigs() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("reference.conf", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("customMerge"));
    assertThat(callback.dates).isEqualTo(Arrays.asList(DOS_EPOCH));
  }

  @Test
  public void testClassInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new DefaultJarEntryFilter().accept("a.class", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("copy"));
    assertThat(callback.dates)
        .isEqualTo(Arrays.asList(DefaultJarEntryFilter.DOS_EPOCH_PLUS_2_SECONDS));
  }

  @Test
  public void testOtherSkippedInputs() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    ZipEntryFilter filter = new DefaultJarEntryFilter();
    filter.accept("a.SF", callback);
    filter.accept("a.DSA", callback);
    filter.accept("a.RSA", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("skip", "skip", "skip"));
    assertThat(callback.dates).isEqualTo(Arrays.<Date>asList());
  }
}
