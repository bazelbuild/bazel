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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.cache.CompactPersistentActionCache;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixGlob;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

/**
 * These tests belong to {@link TimestampBuilderTest}, but they're in a
 * separate class for now because they are a little slower.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class TimestampBuilderMediumTest extends TimestampBuilderTestCase {
  private Path cacheRoot;
  private CompactPersistentActionCache cache;

  @Before
  public final void setCache() throws Exception  {
    // BlazeRuntime.setupLogging(Level.FINEST);  // Uncomment this for debugging.

    cacheRoot = scratch.dir("cacheRoot");
    cache = createCache();
  }

  private CompactPersistentActionCache createCache() throws IOException {
    return new CompactPersistentActionCache(cacheRoot, clock);
  }

  /**
   * Creates and returns a new caching builder based on a given {@code cache}.
   */
  private Builder persistentBuilder(CompactPersistentActionCache cache) throws Exception {
    return createBuilder(cache);
  }

  // TODO(blaze-team): (2009) :
  // - test timestamp monotonicity is not required (i.e. set mtime backwards)
  // - test change of key causes rebuild

  @Test
  public void testUnneededInputs() throws Exception {
    Artifact hello = createSourceArtifact("hello");
    BlazeTestUtils.makeEmptyFile(hello.getPath());
    Artifact optional = createSourceArtifact("hello.optional");
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(Sets.newHashSet(hello, optional), Sets.newHashSet(goodbye));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    // Creating a new persistent cache does not cause a rebuild
    cache.save();
    cache = createCache();
    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt

    BlazeTestUtils.makeEmptyFile(optional.getPath());

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    optional.getPath().delete();

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    // Creating a new persistent cache does not cause a rebuild
    cache.save();
    cache = createCache();
    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt
  }

  @Test
  public void testPersistentCache_ModifyingInputCausesActionReexecution() throws Exception {
    // /hello -> [action] -> /goodbye
    Artifact hello = createSourceArtifact("hello");
    BlazeTestUtils.makeEmptyFile(hello.getPath());
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(Sets.newHashSet(hello), Sets.newHashSet(goodbye));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    FileSystemUtils.touchFile(hello.getPath());

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // rebuilt

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    // Creating a new persistent cache does not cause a rebuild
    cache.save();
    buildArtifacts(persistentBuilder(createCache()), goodbye);
    assertFalse(button.pressed); // not rebuilt
  }

  @Test
  public void testModifyingInputCausesActionReexecution() throws Exception {
    // /hello -> [action] -> /goodbye
    Artifact hello = createSourceArtifact("hello");
    FileSystemUtils.createDirectoryAndParents(hello.getPath().getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content1");
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(Sets.newHashSet(hello), Sets.newHashSet(goodbye));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // still not rebuilt

    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content2");

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // rebuilt

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    // Creating a new persistent cache does not cause a rebuild
    cache.save();
    buildArtifacts(persistentBuilder(createCache()), goodbye);
    assertFalse(button.pressed); // not rebuilt
  }

  @Test
  public void testArtifactOrderingDoesNotMatter() throws Exception {
    // (/hello,/there) -> [action] -> /goodbye

    Artifact hello = createSourceArtifact("hello");
    Artifact there = createSourceArtifact("there");
    FileSystemUtils.createDirectoryAndParents(hello.getPath().getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "hello");
    FileSystemUtils.writeContentAsLatin1(there.getPath(), "there");
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button =
        createActionButton(
            Sets.newLinkedHashSet(ImmutableList.of(hello, there)), Sets.newHashSet(goodbye));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    // Now create duplicate graph, with swapped order.
    clearActions();
    Artifact goodbye2 = createDerivedArtifact("goodbye");
    Button button2 =
        createActionButton(
            Sets.newLinkedHashSet(ImmutableList.of(there, hello)), Sets.newHashSet(goodbye2));

    button2.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button2.pressed); // still not rebuilt
  }

  @Test
  public void testOldCacheKeysAreCleanedUp() throws Exception {
    // [action1] -> (/goodbye), cache key will be /goodbye
    Artifact goodbye = createDerivedArtifact("goodbye");
    FileSystemUtils.createDirectoryAndParents(goodbye.getPath().getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(goodbye.getPath(), "test");
    Button button = createActionButton(emptySet, Sets.newLinkedHashSet(ImmutableList.of(goodbye)));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    // action1 is cached using the cache key /goodbye.
    assertThat(cache.get(goodbye.getExecPathString())).isNotNull();

    // [action2] -> (/hello,/goodbye), cache key will be /hello
    clearActions();
    Artifact hello = createDerivedArtifact("hello");
    Artifact goodbye2 = createDerivedArtifact("goodbye");
    Button button2 =
        createActionButton(emptySet, Sets.newLinkedHashSet(ImmutableList.of(hello, goodbye2)));

    button2.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello, goodbye2);
    assertTrue(button2.pressed); // rebuilt

    // action2 is cached using the cache key /hello.
    assertThat(cache.get(hello.getExecPathString())).isNotNull();

    // Now, action1 should no longer be in the cache.
    assertThat(cache.get(goodbye.getExecPathString())).isNull();
  }

  @Test
  public void testArtifactNamesMatter() throws Exception {
    // /hello -> [action] -> /goodbye

    Artifact hello = createSourceArtifact("hello");
    FileSystemUtils.createDirectoryAndParents(hello.getPath().getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "hello");
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(Sets.newHashSet(hello), Sets.newHashSet(goodbye));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    // Now create duplicate graph, replacing "hello" with "hi".
    clearActions();
    Artifact hi = createSourceArtifact("hi");
    FileSystemUtils.createDirectoryAndParents(hi.getPath().getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(hi.getPath(), "hello");
    Artifact goodbye2 = createDerivedArtifact("goodbye");
    Button button2 = createActionButton(Sets.newHashSet(hi), Sets.newHashSet(goodbye2));

    button2.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye2);
    assertTrue(button2.pressed); // name changed. must rebuild.
  }

  @Test
  public void testDuplicateInputs() throws Exception {
    // (/hello,/hello) -> [action] -> /goodbye

    Artifact hello = createSourceArtifact("hello");
    FileSystemUtils.createDirectoryAndParents(hello.getPath().getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "hello");
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button =
        createActionButton(Lists.<Artifact>newArrayList(hello, hello), Sets.newHashSet(goodbye));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "hello2");

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // rebuilt

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    // Creating a new persistent cache does not cause a rebuild
    cache.save();
    buildArtifacts(persistentBuilder(createCache()), goodbye);
    assertFalse(button.pressed); // not rebuilt
  }

  /**
   * Tests that changing timestamp of the input file without changing it content
   * does not cause action reexecution when metadata cache uses file digests in
   * addition to the timestamp.
   */
  @Test
  public void testModifyingTimestampOnlyDoesNotCauseActionReexecution() throws Exception {
    // /hello -> [action] -> /goodbye
    Artifact hello = createSourceArtifact("hello");
    FileSystemUtils.createDirectoryAndParents(hello.getPath().getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content1");
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(Sets.newHashSet(hello), Sets.newHashSet(goodbye));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), goodbye);
    assertFalse(button.pressed); // not rebuilt

    // Creating a new persistent caches, including metadata cache does not cause
    // a rebuild
    cache.save();
    Builder builder = persistentBuilder(createCache());
    buildArtifacts(builder, goodbye);
    assertFalse(button.pressed); // not rebuilt
  }

  @Test
  public void testPersistentCache_ModifyingOutputCausesActionReexecution() throws Exception {
    // [action] -> /hello
    Artifact hello = createDerivedArtifact("hello");
    Button button = createActionButton(emptySet, Sets.newHashSet(hello));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt

    BlazeTestUtils.changeModtime(hello.getPath());

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertTrue(button.pressed); // rebuilt

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt

    // Creating a new persistent cache does not cause a rebuild
    cache.save();
    buildArtifacts(persistentBuilder(createCache()), hello);
    assertFalse(button.pressed); // not rebuilt
  }

  @Test
  public void testPersistentCache_missingFilenameIndexCausesActionReexecution() throws Exception {
    // [action] -> /hello
    Artifact hello = createDerivedArtifact("hello");
    Button button = createActionButton(emptySet, Sets.newHashSet(hello));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt

    BlazeTestUtils.changeModtime(hello.getPath());

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertTrue(button.pressed); // rebuilt

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt

    // Creating a new persistent cache does not cause a rebuild
    cache.save();

    // Remove filename index file.
    assertTrue(
        Iterables.getOnlyElement(
                UnixGlob.forPath(cacheRoot).addPattern("filename_index*").globInterruptible())
            .delete());

    // Now first cache creation attempt should cause IOException while renaming corrupted files.
    // Second attempt will initialize empty cache, causing rebuild.
    try {
      createCache();
      fail("Expected IOException");
    } catch (IOException e) {
      assertThat(e).hasMessage("Failed action cache referential integrity check: empty index");
    }

    buildArtifacts(persistentBuilder(createCache()), hello);
    assertTrue(button.pressed); // rebuilt due to the missing filename index
  }

  @Test
  public void testPersistentCache_failedIntegrityCheckCausesActionReexecution() throws Exception {
    // [action] -> /hello
    Artifact hello = createDerivedArtifact("hello");
    Button button = createActionButton(emptySet, Sets.newHashSet(hello));

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertTrue(button.pressed); // built

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt

    BlazeTestUtils.changeModtime(hello.getPath());

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertTrue(button.pressed); // rebuilt

    button.pressed = false;
    buildArtifacts(persistentBuilder(cache), hello);
    assertFalse(button.pressed); // not rebuilt

    cache.save();

    // Get filename index path and store a copy of it.
    Path indexPath =
        Iterables.getOnlyElement(
            UnixGlob.forPath(cacheRoot).addPattern("filename_index*").globInterruptible());
    Path indexCopy = scratch.resolve("index_copy");
    FileSystemUtils.copyFile(indexPath, indexCopy);

    // Add extra records to the action cache and indexer.
    Artifact helloExtra = createDerivedArtifact("hello_extra");
    Button buttonExtra = createActionButton(emptySet, Sets.newHashSet(helloExtra));
    buildArtifacts(persistentBuilder(cache), helloExtra);
    assertTrue(buttonExtra.pressed); // built

    cache.save();
    assertTrue(indexPath.getFileSize() > indexCopy.getFileSize());

    // Validate current cache.
    buildArtifacts(persistentBuilder(createCache()), hello);
    assertFalse(button.pressed); // not rebuilt

    // Restore outdated file index.
    FileSystemUtils.copyFile(indexCopy, indexPath);

    // Now first cache creation attempt should cause IOException while renaming corrupted files.
    // Second attempt will initialize empty cache, causing rebuild.
    try {
      createCache();
      fail("Expected IOException");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("Failed action cache referential integrity check");
    }

    // Validate cache with incorrect (out-of-date) filename index.
    buildArtifacts(persistentBuilder(createCache()), hello);
    assertTrue(button.pressed); // rebuilt due to the out-of-date index
  }
}
