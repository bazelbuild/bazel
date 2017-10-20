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

import com.android.dex.Dex;
import com.android.dx.command.dexer.DxContext;
import com.android.dx.dex.DexOptions;
import com.android.dx.dex.cf.CfOptions;
import com.android.dx.dex.file.DexFile;
import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.io.InputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DexLimitTracker}. */
@RunWith(JUnit4.class)
public class DexLimitTrackerTest {

  private Dex dex;

  @Before
  public void setUp() throws IOException {
    dex = DexFiles.toDex(convertClass(DexLimitTrackerTest.class));
  }

  @Test
  public void testUnderLimit() {
    DexLimitTracker tracker =
        new DexLimitTracker(Math.max(dex.methodIds().size(), dex.fieldIds().size()));
    assertThat(tracker.track(dex)).isFalse();
  }

  @Test
  public void testOverLimit() throws IOException {
    DexLimitTracker tracker =
        new DexLimitTracker(Math.max(dex.methodIds().size(), dex.fieldIds().size()) - 1);
    assertThat(tracker.track(dex)).isTrue();
    assertThat(tracker.track(dex)).isTrue();
    assertThat(tracker.track(DexFiles.toDex(convertClass(DexLimitTracker.class)))).isTrue();
  }

  @Test
  public void testRepeatedReferencesDeduped() throws IOException {
    DexLimitTracker tracker =
        new DexLimitTracker(Math.max(dex.methodIds().size(), dex.fieldIds().size()));
    assertThat(tracker.track(dex)).isFalse();
    assertThat(tracker.track(dex)).isFalse();
    assertThat(tracker.track(dex)).isFalse();
    assertThat(tracker.track(dex)).isFalse();
    assertThat(tracker.track(DexFiles.toDex(convertClass(DexLimitTracker.class)))).isTrue();
    assertThat(tracker.track(dex)).isTrue();
  }

  @Test
  public void testGoOverLimit() throws IOException {
    DexLimitTracker tracker =
        new DexLimitTracker(Math.max(dex.methodIds().size(), dex.fieldIds().size()));
    assertThat(tracker.track(dex)).isFalse();
    assertThat(tracker.track(DexFiles.toDex(convertClass(DexLimitTracker.class)))).isTrue();
  }

  @Test
  public void testClear() throws IOException {
    DexLimitTracker tracker =
        new DexLimitTracker(Math.max(dex.methodIds().size(), dex.fieldIds().size()));
    assertThat(tracker.track(dex)).isFalse();
    assertThat(tracker.track(DexFiles.toDex(convertClass(DexLimitTracker.class)))).isTrue();
    tracker.clear();
    assertThat(tracker.track(dex)).isFalse();
  }

  private static DexFile convertClass(Class<?> clazz) throws IOException {
    String path = clazz.getName().replace('.', '/') + ".class";
    try (InputStream in =
        Thread.currentThread().getContextClassLoader().getResourceAsStream(path)) {
      return new DexConverter(new Dexing(new DxContext(), new DexOptions(), new CfOptions()))
          .toDexFile(ByteStreams.toByteArray(in), path);
    }
  }
}
