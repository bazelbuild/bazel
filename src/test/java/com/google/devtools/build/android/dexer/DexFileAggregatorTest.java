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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.android.dex.Dex;
import com.android.dx.command.dexer.DxContext;
import com.android.dx.dex.DexOptions;
import com.android.dx.dex.cf.CfOptions;
import com.android.dx.dex.file.DexFile;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipEntry;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link DexFileAggregator}. */
@RunWith(JUnit4.class)
public class DexFileAggregatorTest {

  /** Standard .dex file limit on methods and fields. */
  private static final int DEX_LIMIT = 265 * 265;
  private static final int WASTE = 1;

  @Mock private DexFileArchive dest;
  @Captor private ArgumentCaptor<Dex> written;

  private Dex dex;

  @Before
  public void setUp() throws IOException {
    dex = DexFiles.toDex(convertClass(DexFileAggregatorTest.class));
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testClose_emptyWritesNothing() throws Exception {
    DexFileAggregator dexer =
        new DexFileAggregator(
            new DxContext(),
            dest,
            newDirectExecutorService(),
            MultidexStrategy.MINIMAL,
            /*forceJumbo=*/ false,
            DEX_LIMIT,
            WASTE,
            DexFileMergerTest.DEX_PREFIX);
    dexer.close();
    verify(dest, times(0)).addFile(any(ZipEntry.class), any(Dex.class));
  }

  @Test
  public void testAddAndClose_singleInputWritesThatInput() throws Exception {
    DexFileAggregator dexer =
        new DexFileAggregator(
            new DxContext(),
            dest,
            newDirectExecutorService(),
            MultidexStrategy.MINIMAL,
            /*forceJumbo=*/ false,
            0,
            WASTE,
            DexFileMergerTest.DEX_PREFIX);
    dexer.add(dex);
    dexer.close();
    verify(dest).addFile(any(ZipEntry.class), eq(dex));
  }

  @Test
  public void testAddAndClose_forceJumboRewrites() throws Exception {
    DexFileAggregator dexer =
        new DexFileAggregator(
            new DxContext(),
            dest,
            newDirectExecutorService(),
            MultidexStrategy.MINIMAL,
            /*forceJumbo=*/ true,
            0,
            WASTE,
            DexFileMergerTest.DEX_PREFIX);
    dexer.add(dex);
    try {
      dexer.close();
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().isEqualTo("--forceJumbo flag not supported");
      System.err.println("Skipping this test due to missing --forceJumbo support in Android SDK.");
      e.printStackTrace();
      return;
    }

    verify(dest).addFile(any(ZipEntry.class), written.capture());
    assertThat(written.getValue()).isNotEqualTo(dex);
    assertThat(written.getValue().getLength()).isGreaterThan(dex.getLength());
  }

  @Test
  public void testMultidex_underLimitWritesOneShard() throws Exception {
    DexFileAggregator dexer =
        new DexFileAggregator(
            new DxContext(),
            dest,
            newDirectExecutorService(),
            MultidexStrategy.BEST_EFFORT,
            /*forceJumbo=*/ false,
            DEX_LIMIT,
            WASTE,
            DexFileMergerTest.DEX_PREFIX);
    Dex dex2 = DexFiles.toDex(convertClass(ByteStreams.class));
    dexer.add(dex);
    dexer.add(dex2);
    verify(dest, times(0)).addFile(any(ZipEntry.class), any(Dex.class));
    dexer.close();
    verify(dest).addFile(any(ZipEntry.class), written.capture());
    assertThat(Iterables.size(written.getValue().classDefs())).isEqualTo(2);
  }

  @Test
  public void testMultidex_overLimitWritesSecondShard() throws Exception {
    DexFileAggregator dexer =
        new DexFileAggregator(
            new DxContext(),
            dest,
            newDirectExecutorService(),
            MultidexStrategy.BEST_EFFORT,
            /*forceJumbo=*/ false,
            2 /* dex has more than 2 methods and fields */,
            WASTE,
            DexFileMergerTest.DEX_PREFIX);
    Dex dex2 = DexFiles.toDex(convertClass(ByteStreams.class));
    dexer.add(dex);   // classFile is already over limit but we take anything in empty shard
    dexer.add(dex2);  // this should start a new shard
    // Make sure there was one file written and that file is dex
    verify(dest).addFile(any(ZipEntry.class), written.capture());
    assertThat(written.getValue()).isSameInstanceAs(dex);
    dexer.close();
    verify(dest).addFile(any(ZipEntry.class), eq(dex2));
  }

  @Test
  public void testMonodex_alwaysWritesSingleShard() throws Exception {
    DexFileAggregator dexer =
        new DexFileAggregator(
            new DxContext(),
            dest,
            newDirectExecutorService(),
            MultidexStrategy.OFF,
            /*forceJumbo=*/ false,
            2 /* dex has more than 2 methods and fields */,
            WASTE,
            DexFileMergerTest.DEX_PREFIX);
    Dex dex2 = DexFiles.toDex(convertClass(ByteStreams.class));
    dexer.add(dex);
    dexer.add(dex2);
    verify(dest, times(0)).addFile(any(ZipEntry.class), any(Dex.class));
    dexer.close();
    verify(dest).addFile(any(ZipEntry.class), written.capture());
    assertThat(Iterables.size(written.getValue().classDefs())).isEqualTo(2);
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
