package com.google.devtools.build.lib.bazel.repository.decompressor;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link DecompressorDescriptor}. */
@RunWith(JUnit4.class)
public class DecompressorDescriptorTest {

  /**
   * Returns a {#link DecompressorDescriptor} with the given includes and excludes. Stubs all other
   * required values.
   */
  public static DecompressorDescriptor inexDescriptor(
      List<String> includes, List<String> excludes) {
    FileSystem testFs = FileSystems.getNativeFileSystem();

    return DecompressorDescriptor.builder()
        .setIncludes(includes)
        .setExcludes(excludes)
        .setDestinationPath(testFs.getPath("/stubOutPath"))
        .setArchivePath(testFs.getPath("/stubArchivePath"))
        .build();
  }

  /** Asserts that a given archiveEntry should have been skipped. */
  public static void assertSkipped(DecompressorDescriptor d, String archiveEntry) {
    assertTrue(
        String.format(
            "DecompressorDescriptor includes: '%s' and excludes: '%s', but archive entry '%s' was NOT skipped",
            d.includes().isEmpty() ? "[]" : String.join(", ", d.includes()),
            d.excludes().isEmpty() ? "[]" : String.join(", ", d.excludes()),
            archiveEntry),
        d.skipArchiveEntry(archiveEntry));
  }

  /** Asserts that a given archiveEntry should have been processed. */
  public static void assertProcessed(DecompressorDescriptor d, String archiveEntry) {
    assertFalse(
        String.format(
            "DecompressorDescriptor includes: '%s' and excludes: '%s', but archive entry '%s' was NOT processed",
            d.includes().isEmpty() ? "[]" : String.join(", ", d.includes()),
            d.excludes().isEmpty() ? "[]" : String.join(", ", d.excludes()),
            archiveEntry),
        d.skipArchiveEntry(archiveEntry));
  }

  @Test
  public void testEmptyIncludesSkipsNothing() {
    DecompressorDescriptor d =
        inexDescriptor(/* includes= */ ImmutableList.of(), /* excludes= */ ImmutableList.of());
    assertProcessed(d, "anyFileEntry/path");
    assertProcessed(d, "anyFileEntry/path2");
  }

  @Test
  public void testIncludesExclusive() {
    DecompressorDescriptor d =
        inexDescriptor(
            /* includes= */ ImmutableList.of("anyFileEntry**"), /* excludes= */ ImmutableList.of());
    assertProcessed(d, "anyFileEntry/path");
    assertSkipped(d, "excluded/file");
  }

  @Test
  public void testExclude() {
    DecompressorDescriptor d =
        inexDescriptor(
            /* includes= */ ImmutableList.of(),
            /* excludes= */ ImmutableList.of("unneededData/**"));
    assertProcessed(d, "anyFileEntry/path");
    assertSkipped(d, "unneededData/large.file");
  }

  @Test
  public void testExcludeTakesPrecedenceOverIncludes() {
    DecompressorDescriptor d =
        inexDescriptor(
            /* includes= */ ImmutableList.of("anyFileEntry**"),
            /* excludes= */ ImmutableList.of("anyFileEntry/**.exclude"));
    assertProcessed(d, "anyFileEntry/path");
    assertSkipped(d, "anyFileEntry/path2.exclude");
  }

  @Test
  public void testMultipleIncludesExcludes() {
    DecompressorDescriptor d =
        inexDescriptor(
            /* includes= */ ImmutableList.of("a/**", "b/**", "c/**"),
            /* excludes= */ ImmutableList.of("b/**.exclude", "a/exclude"));
    assertProcessed(d, "a/file/path");
    assertSkipped(d, "a/exclude");
    assertProcessed(d, "b/file/path");
    assertSkipped(d, "b/file/path.exclude");
    assertProcessed(d, "c/file/path");
    assertSkipped(d, "d/file/path");
  }
}
