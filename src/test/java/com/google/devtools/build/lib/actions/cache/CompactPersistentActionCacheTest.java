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
package com.google.devtools.build.lib.actions.cache;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.time.Instant.EPOCH;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.ActionCache.Entry.SerializableTreeArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for the CompactPersistentActionCache class. */
@RunWith(TestParameterInjector.class)
public class CompactPersistentActionCacheTest {

  private final Scratch scratch = new Scratch();
  private Path execRoot;
  private Path cacheRoot;
  private Path corruptedCacheRoot;
  private Path tmpDir;
  private Path mapFile;
  private Path journalFile;
  private Path indexFile;
  private Path indexJournalFile;
  private Path timestampFile;
  private Path timestampJournalFile;
  private final ManualClock clock = new ManualClock();
  private CompactPersistentActionCache cache;
  private ArtifactRoot artifactRoot;

  private final EventHandler eventHandler = spy(EventHandler.class);

  @Before
  public final void createFiles() throws Exception  {
    execRoot = scratch.resolve("/output");
    cacheRoot = scratch.resolve("/cache_root");
    corruptedCacheRoot = scratch.resolve("/corrupted_cache_root");
    tmpDir = scratch.resolve("/cache_tmp_dir");
    cache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, NullEventHandler.INSTANCE);
    mapFile = CompactPersistentActionCache.cacheFile(cacheRoot);
    journalFile = CompactPersistentActionCache.journalFile(cacheRoot);
    indexFile = CompactPersistentActionCache.indexFile(cacheRoot);
    indexJournalFile = CompactPersistentActionCache.indexJournalFile(cacheRoot);
    timestampFile = CompactPersistentActionCache.timestampFile(cacheRoot);
    timestampJournalFile = CompactPersistentActionCache.timestampJournalFile(cacheRoot);
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, ArtifactRoot.RootType.OUTPUT, "bin");
  }

  @Test
  public void testGetInvalidKey() {
    assertThat(cache.get("key")).isNull();
  }

  @Test
  public void testPutAndGet() {
    String key = "key";
    putKey(key);
    ActionCache.Entry readentry = cache.get(key);
    assertThat(readentry).isNotNull();
    assertThat(readentry.toString()).isEqualTo(cache.get(key).toString());
    assertThat(mapFile.exists()).isFalse();
  }

  @Test
  public void testPutAndRemove() {
    String key = "key";
    putKey(key);
    cache.remove(key);
    assertThat(cache.get(key)).isNull();
    assertThat(mapFile.exists()).isFalse();
  }

  @Test
  public void testGetSize() {
    // initial state.
    assertThat(cache.size()).isEqualTo(0);

    String key = "key";
    putKey(key);
    // the inserted key, and the validation key
    assertThat(cache.size()).isEqualTo(2);

    cache.remove(key);
    // the validation key
    assertThat(cache.size()).isEqualTo(1);

    cache.clear();
    assertThat(cache.size()).isEqualTo(0);
  }

  @Test
  public void testSaveDiscoverInputs() throws Exception {
    assertSave(true);
  }

  @Test
  public void testSaveNoDiscoverInputs() throws Exception {
    assertSave(false);
  }

  private void assertSave(boolean discoverInputs) throws Exception {
    String key = "key";
    putKey(key, discoverInputs);
    cache.save();
    assertThat(mapFile.exists()).isTrue();
    assertThat(journalFile.exists()).isFalse();

    CompactPersistentActionCache newCache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);
    verify(eventHandler, never()).handle(any());
    ActionCache.Entry readentry = newCache.get(key);
    assertThat(readentry).isNotNull();
    assertThat(readentry.toString()).isEqualTo(cache.get(key).toString());
  }

  @Test
  public void testIncrementalSave() throws IOException {
    for (int i = 0; i < 300; i++) {
      putKey(Integer.toString(i));
    }
    assertFullSave();

    // Add 2 entries to 300. Might as well just leave them in the journal.
    putKey("abc");
    putKey("123");
    assertIncrementalSave(cache);

    // Make sure we have all the entries, including those in the journal,
    // after deserializing into a new cache.
    CompactPersistentActionCache newcache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);
    verify(eventHandler, never()).handle(any());
    for (int i = 0; i < 100; i++) {
      assertKeyEquals(cache, newcache, Integer.toString(i));
    }
    assertKeyEquals(cache, newcache, "abc");
    assertKeyEquals(cache, newcache, "123");
    putKey("xyz", newcache, true);
    assertIncrementalSave(newcache);

    // Make sure we can see previous journal values after a second incremental save.
    CompactPersistentActionCache newerCache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);
    verify(eventHandler, never()).handle(any());
    for (int i = 0; i < 100; i++) {
      assertKeyEquals(cache, newerCache, Integer.toString(i));
    }
    assertKeyEquals(cache, newerCache, "abc");
    assertKeyEquals(cache, newerCache, "123");
    assertThat(newerCache.get("xyz")).isNotNull();
    assertThat(newerCache.get("not_a_key")).isNull();

    // Add another 10 entries. This should not be incremental.
    for (int i = 300; i < 310; i++) {
      putKey(Integer.toString(i));
    }
    assertFullSave();
  }

  @Test
  public void testRemoveIf() throws IOException {
    // Add 100 entries, 5 of which discover inputs, and do a full save.
    for (int i = 0; i < 100; i++) {
      putKey(Integer.toString(i), i % 20 == 0);
    }
    assertFullSave();

    // Remove entries that discover inputs and flush the journal.
    cache.removeIf(e -> e.discoversInputs());
    assertIncrementalSave(cache);

    // Check that the entries that discover inputs are gone, and the rest are still there.
    for (int i = 0; i < 100; i++) {
      ActionCache.Entry entry = cache.get(Integer.toString(i));
      if (i % 20 == 0) {
        assertThat(entry).isNull();
      } else {
        assertThat(entry).isNotNull();
      }
    }

    // Make sure we get the same result after deserializing into a new cache.
    CompactPersistentActionCache newerCache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);
    verify(eventHandler, never()).handle(any());
    for (int i = 0; i < 100; i++) {
      ActionCache.Entry entry = newerCache.get(Integer.toString(i));
      if (i % 20 == 0) {
        assertThat(entry).isNull();
      } else {
        assertThat(entry).isNotNull();
      }
    }
  }

  @Test
  public void testClear() throws IOException {
    // Add 100 entries and do a full save.
    for (int i = 0; i < 100; i++) {
      putKey(Integer.toString(i));
    }
    assertFullSave();

    // Clear the cache (which implicitly saves it).
    cache.clear();

    // Check that the cache is empty.
    assertThat(cache.size()).isEqualTo(0);

    // Make sure we get the same result after deserializing into a new cache.
    CompactPersistentActionCache newerCache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);
    verify(eventHandler, never()).handle(any());
    assertThat(newerCache.size()).isEqualTo(0);
  }

  @Test
  public void testTimestamps() throws IOException {
    clock.advance(Duration.ofDays(100));
    putKey("abc");
    clock.advance(Duration.ofDays(100));
    putKey("def");
    clock.advance(Duration.ofDays(100));
    putKey("ghi");
    clock.advance(Duration.ofDays(100));
    putKey("jkl");
    clock.advance(Duration.ofDays(100));
    putKey("mno", /* discoversInputs= */ true);

    // Getting an entry should update its timestamp.
    clock.advance(Duration.ofDays(100));
    var unused = cache.get("abc");

    // Overwriting an entry should update its timestamp.
    clock.advance(Duration.ofDays(100));
    putKey("def");

    // Remove an entry should remove its timestamp.
    clock.advance(Duration.ofDays(100));
    cache.remove("ghi");

    // Removing entries matching a predicate should not affect the timestamp of other entries.
    clock.advance(Duration.ofDays(100));
    cache.removeIf(e -> e.discoversInputs());

    assertFullSave();

    assertThat(cache.getActionTimestampMap())
        .containsExactly(
            "abc",
            EPOCH.plus(Duration.ofDays(600)),
            "def",
            EPOCH.plus(Duration.ofDays(700)),
            "jkl",
            EPOCH.plus(Duration.ofDays(400)));

    // Make sure we get the same result after deserializing into a new cache.
    CompactPersistentActionCache newerCache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);
    verify(eventHandler, never()).handle(any());
    assertThat(newerCache.getActionTimestampMap())
        .containsExactly(
            "abc",
            EPOCH.plus(Duration.ofDays(600)),
            "def",
            EPOCH.plus(Duration.ofDays(700)),
            "jkl",
            EPOCH.plus(Duration.ofDays(400)));
  }

  @Test
  public void testTrimNoThreshold() throws Exception {
    clock.advance(Duration.ofDays(100));
    putKey("abc");
    clock.advance(Duration.ofDays(100));
    putKey("def");
    clock.advance(Duration.ofDays(100));
    putKey("ghi");
    clock.advance(Duration.ofDays(100));
    putKey("jkl");
    clock.advance(Duration.ofDays(100));
    assertFullSave();

    cache = cache.trim(0, Duration.ofDays(250));

    // Check that the cache was trimmed correctly.
    assertThat(cache.get("abc")).isNull();
    assertThat(cache.get("def")).isNull();
    assertThat(cache.get("ghi")).isNotNull();
    assertThat(cache.get("jkl")).isNotNull();

    // Make sure we get the same result after deserializing into a new cache.
    CompactPersistentActionCache newerCache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);
    verify(eventHandler, never()).handle(any());
    assertThat(newerCache.get("abc")).isNull();
    assertThat(newerCache.get("def")).isNull();
    assertThat(newerCache.get("ghi")).isNotNull();
    assertThat(newerCache.get("jkl")).isNotNull();
  }

  @Test
  public void testTrimBelowThreshold() throws Exception {
    clock.advance(Duration.ofDays(100));
    putKey("abc");
    clock.advance(Duration.ofDays(100));
    putKey("def");
    clock.advance(Duration.ofDays(100));
    putKey("ghi");
    clock.advance(Duration.ofDays(100));
    putKey("jkl");
    clock.advance(Duration.ofDays(100));
    assertFullSave();

    // 1 of 4 entries is stale, below 30% threshold.
    cache = cache.trim(0.3f, Duration.ofDays(350));

    assertThat(cache.get("abc")).isNotNull();
    assertThat(cache.get("def")).isNotNull();
    assertThat(cache.get("ghi")).isNotNull();
    assertThat(cache.get("jkl")).isNotNull();
  }

  @Test
  public void testTrimAboveThreshold() throws Exception {
    clock.advance(Duration.ofDays(100));
    putKey("abc");
    clock.advance(Duration.ofDays(100));
    putKey("def");
    clock.advance(Duration.ofDays(100));
    putKey("ghi");
    clock.advance(Duration.ofDays(100));
    putKey("jkl");
    clock.advance(Duration.ofDays(100));
    assertFullSave();

    // 1 of 4 entries is stale, above 20% threshold.
    cache = cache.trim(0.2f, Duration.ofDays(350));

    assertThat(cache.get("abc")).isNull();
    assertThat(cache.get("def")).isNotNull();
    assertThat(cache.get("ghi")).isNotNull();
    assertThat(cache.get("jkl")).isNotNull();
  }

  enum IncompatibleFile {
    MAP_FILE,
    JOURNAL_FILE,
    INDEX_FILE,
    INDEX_JOURNAL_FILE,
    TIMESTAMP_FILE,
    TIMESTAMP_JOURNAL_FILE
  }

  @Test
  public void testIncompatibleFormat(@TestParameter IncompatibleFile param) throws IOException {
    Path incompatibleFile =
        switch (param) {
          case MAP_FILE -> mapFile;
          case JOURNAL_FILE -> journalFile;
          case INDEX_FILE -> indexFile;
          case INDEX_JOURNAL_FILE -> indexJournalFile;
          case TIMESTAMP_FILE -> timestampFile;
          case TIMESTAMP_JOURNAL_FILE -> timestampJournalFile;
        };

    FileSystemUtils.writeContent(incompatibleFile, "incompatible".getBytes(ISO_8859_1));

    cache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);

    verify(eventHandler, never()).handle(any());
    assertThat(corruptedCacheRoot.exists()).isFalse();
    assertThat(cache.size()).isEqualTo(0);
  }

  @Test
  public void testTruncatedMapFile() throws IOException {
    for (int i = 0; i < 300; i++) {
      putKey(Integer.toString(i));
    }
    assertFullSave();

    byte[] contents = FileSystemUtils.readContent(mapFile);
    FileSystemUtils.writeContent(mapFile, Arrays.copyOf(contents, contents.length - 1));

    cache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);

    verify(eventHandler).handle(any());
    assertThat(corruptedCacheRoot.exists()).isTrue();
    assertThat(cache.size()).isEqualTo(0);
  }

  @Test
  public void testTruncatedJournalFile() throws IOException {
    for (int i = 0; i < 300; i++) {
      putKey(Integer.toString(i));
    }
    assertFullSave();

    putKey("abc");
    assertIncrementalSave(cache);

    assertThat(cache.size()).isEqualTo(302); // 301 entries + validation record

    byte[] contents = FileSystemUtils.readContent(journalFile);
    FileSystemUtils.writeContent(journalFile, Arrays.copyOf(contents, contents.length - 1));

    cache =
        CompactPersistentActionCache.create(
            cacheRoot, corruptedCacheRoot, tmpDir, clock, eventHandler);

    verify(eventHandler, never()).handle(any());
    assertThat(corruptedCacheRoot.exists()).isFalse();
    assertThat(cache.size()).isEqualTo(301);
  }

  private FileArtifactValue createLocalMetadata(Artifact artifact, String content)
      throws IOException {
    artifact.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(artifact.getPath(), content);
    return FileArtifactValue.createForTesting(artifact.getPath());
  }

  private FileArtifactValue createRemoteMetadata(
      Artifact artifact,
      String content,
      @Nullable Instant expirationTime,
      @Nullable PathFragment resolvedPath) {
    byte[] bytes = content.getBytes(StandardCharsets.UTF_8);
    byte[] digest =
        artifact
            .getPath()
            .getFileSystem()
            .getDigestFunction()
            .getHashFunction()
            .hashBytes(bytes)
            .asBytes();
    FileArtifactValue metadata =
        FileArtifactValue.createForRemoteFileWithMaterializationData(
            digest, bytes.length, 1, expirationTime);
    if (resolvedPath != null) {
      metadata = FileArtifactValue.createFromExistingWithResolvedPath(metadata, resolvedPath);
    }
    return metadata;
  }

  private FileArtifactValue createRemoteMetadata(
      Artifact artifact, String content, @Nullable PathFragment resolvedPath) {
    return createRemoteMetadata(artifact, content, /* expirationTime= */ null, resolvedPath);
  }

  private FileArtifactValue createRemoteMetadata(Artifact artifact, String content) {
    return createRemoteMetadata(artifact, content, /* resolvedPath= */ null);
  }

  private TreeArtifactValue createTreeMetadata(
      SpecialArtifact parent,
      ImmutableMap<String, FileArtifactValue> children,
      Optional<FileArtifactValue> archivedArtifactValue,
      Optional<PathFragment> resolvedPath) {
    TreeArtifactValue.Builder builder = TreeArtifactValue.newBuilder(parent);
    for (Map.Entry<String, FileArtifactValue> entry : children.entrySet()) {
      builder.putChild(
          Artifact.TreeFileArtifact.createTreeOutput(parent, entry.getKey()), entry.getValue());
    }
    archivedArtifactValue.ifPresent(
        metadata -> {
          ArchivedTreeArtifact artifact = ArchivedTreeArtifact.createForTree(parent);
          builder.setArchivedRepresentation(
              TreeArtifactValue.ArchivedRepresentation.create(artifact, metadata));
        });
    if (resolvedPath.isPresent()) {
      builder.setResolvedPath(resolvedPath.get());
    }
    return builder.build();
  }

  @Test
  public void putAndGet_savesRemoteFileMetadata() {
    Artifact artifact = ActionsTestUtil.DUMMY_ARTIFACT;
    FileArtifactValue metadata = createRemoteMetadata(artifact, "content");
    var entry =
        builder("key").addOutputFile(artifact, metadata, /* saveFileMetadata= */ true).build();
    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputFile(artifact)).isEqualTo(metadata);
  }

  @Test
  public void putAndGet_savesRemoteFileMetadata_withExpirationTime() {
    Artifact artifact = ActionsTestUtil.DUMMY_ARTIFACT;
    Instant expirationTime = Instant.now().truncatedTo(ChronoUnit.MILLIS);
    FileArtifactValue metadata =
        createRemoteMetadata(artifact, "content", expirationTime, /* resolvedPath= */ null);
    var entry =
        builder("key").addOutputFile(artifact, metadata, /* saveFileMetadata= */ true).build();
    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputFile(artifact).getExpirationTime()).isEqualTo(expirationTime);
  }

  @Test
  public void putAndGet_savesRemoteFileMetadata_withResolvedPath() {
    Artifact artifact = ActionsTestUtil.DUMMY_ARTIFACT;
    FileArtifactValue metadata =
        createRemoteMetadata(artifact, "content", execRoot.getRelative("some/path").asFragment());
    var entry =
        builder("key").addOutputFile(artifact, metadata, /* saveFileMetadata= */ true).build();
    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputFile(artifact)).isEqualTo(metadata);
  }

  @Test
  public void putAndGet_ignoresLocalFileMetadata() throws IOException {
    Artifact artifact = ActionsTestUtil.DUMMY_ARTIFACT;
    FileArtifactValue metadata = createLocalMetadata(artifact, "content");
    var entry =
        builder("key").addOutputFile(artifact, metadata, /* saveFileMetadata= */ true).build();
    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputFile(artifact)).isNull();
  }

  @Test
  public void putAndGet_treeMetadata_onlySavesRemoteFileMetadata() throws IOException {
    SpecialArtifact artifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            artifactRoot, PathFragment.create("bin/dummy"));
    TreeArtifactValue metadata =
        createTreeMetadata(
            artifact,
            ImmutableMap.of(
                "file1",
                    createRemoteMetadata(
                        Artifact.TreeFileArtifact.createTreeOutput(
                            artifact, PathFragment.create("file1")),
                        "content1"),
                "file2",
                    createLocalMetadata(
                        Artifact.TreeFileArtifact.createTreeOutput(
                            artifact, PathFragment.create("file2")),
                        "content2")),
            /* archivedArtifactValue= */ Optional.empty(),
            /* resolvedPath= */ Optional.empty());
    var entry =
        builder("key").addOutputTree(artifact, metadata, /* saveTreeMetadata= */ true).build();
    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputTree(artifact))
        .isEqualTo(
            SerializableTreeArtifactValue.create(
                ImmutableMap.of(
                    "file1",
                    createRemoteMetadata(
                        Artifact.TreeFileArtifact.createTreeOutput(
                            artifact, PathFragment.create("file1")),
                        "content1")),
                /* archivedFileValue= */ Optional.empty(),
                /* resolvedPath= */ Optional.empty()));
  }

  @Test
  public void putAndGet_treeMetadata_savesRemoteArchivedArtifact() {
    SpecialArtifact artifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            artifactRoot, PathFragment.create("bin/dummy"));
    TreeArtifactValue metadata =
        createTreeMetadata(
            artifact,
            ImmutableMap.of(),
            Optional.of(createRemoteMetadata(artifact, "content")),
            /* resolvedPath= */ Optional.empty());
    var entry =
        builder("key").addOutputTree(artifact, metadata, /* saveTreeMetadata= */ true).build();
    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputTree(artifact))
        .isEqualTo(
            SerializableTreeArtifactValue.create(
                ImmutableMap.of(),
                Optional.of(createRemoteMetadata(artifact, "content")),
                Optional.empty()));
  }

  @Test
  public void putAndGet_treeMetadata_ignoresLocalArchivedArtifact() throws IOException {
    SpecialArtifact artifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            artifactRoot, PathFragment.create("bin/dummy"));
    TreeArtifactValue metadata =
        createTreeMetadata(
            artifact,
            ImmutableMap.of(),
            Optional.of(
                createLocalMetadata(
                    ActionsTestUtil.createArtifact(artifactRoot, "bin/archive"), "content")),
            /* resolvedPath= */ Optional.empty());
    var entry =
        builder("key").addOutputTree(artifact, metadata, /* saveTreeMetadata= */ true).build();
    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputTree(artifact)).isNull();
  }

  @Test
  public void putAndGet_treeMetadata_savesResolvedPath() {
    PathFragment resolvedPath = execRoot.getRelative("some/path").asFragment();
    SpecialArtifact artifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            artifactRoot, PathFragment.create("bin/dummy"));
    TreeArtifactValue metadata =
        createTreeMetadata(
            artifact,
            ImmutableMap.of(),
            /* archivedArtifactValue= */ Optional.empty(),
            Optional.of(resolvedPath));
    var entry =
        builder("key").addOutputTree(artifact, metadata, /* saveTreeMetadata= */ true).build();

    cache.put("key", entry);

    entry = cache.get("key");

    assertThat(entry.getOutputTree(artifact))
        .isEqualTo(
            SerializableTreeArtifactValue.create(
                ImmutableMap.of(),
                /* archivedFileValue= */ Optional.empty(),
                Optional.of(resolvedPath)));
  }

  private static void assertKeyEquals(ActionCache cache1, ActionCache cache2, String key) {
    Object entry = cache1.get(key);
    assertThat(entry).isNotNull();
    assertThat(cache2.get(key).toString()).isEqualTo(entry.toString());
  }

  private void assertFullSave() throws IOException {
    cache.save();
    assertThat(mapFile.exists()).isTrue();
    assertThat(journalFile.exists()).isFalse();
  }

  private void assertIncrementalSave(ActionCache ac) throws IOException {
    ac.save();
    assertThat(mapFile.exists()).isTrue();
    assertThat(journalFile.exists()).isTrue();
  }

  private void putKey(String key) {
    putKey(key, cache, false);
  }

  private void putKey(String key, boolean discoversInputs) {
    putKey(key, cache, discoversInputs);
  }

  private void putKey(String key, ActionCache actionCache, boolean discoversInputs) {
    var entry = builder(key, discoversInputs).build();
    actionCache.put(key, entry);
  }

  private static ActionCache.Entry.Builder builder(String actionKey) {
    return builder(actionKey, /* discoversInputs= */ false);
  }

  private static ActionCache.Entry.Builder builder(String actionKey, boolean discoversInputs) {
    return new ActionCache.Entry.Builder(
        actionKey,
        discoversInputs,
        /* clientEnv= */ ImmutableMap.of(),
        /* execProperties= */ ImmutableMap.of(),
        OutputPermissions.READONLY,
        /* useArchivedTreeArtifacts= */ false);
  }
}
