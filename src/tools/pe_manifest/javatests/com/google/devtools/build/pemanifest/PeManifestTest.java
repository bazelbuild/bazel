// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.pemanifest;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link PeManifest}.
 *
 * <p>The test executables were produced by lld-link, see {@code testdata/generate.sh}.
 */
@RunWith(JUnit4.class)
public final class PeManifestTest {

  private static final String TESTDATA =
      "src/tools/pe_manifest/javatests/com/google/devtools/build/pemanifest/testdata/";

  private static final byte[] SMALL_MANIFEST =
      "<assembly manifestVersion=\"1.0\"/>".getBytes(UTF_8);

  @Test
  public void readManifest_withReloc() throws Exception {
    assertThat(PeManifest.readManifest(readTestData("with_reloc.exe")))
        .isEqualTo(readTestData("manifest.xml"));
  }

  @Test
  public void readManifest_rsrcLast() throws Exception {
    assertThat(PeManifest.readManifest(readTestData("rsrc_last.exe")))
        .isEqualTo(readTestData("manifest.xml"));
  }

  @Test
  public void readManifest_pe32() throws Exception {
    assertThat(PeManifest.readManifest(readTestData("pe32.exe")))
        .isEqualTo(readTestData("manifest.xml"));
  }

  @Test
  public void readManifest_noManifest() throws Exception {
    assertThat(PeManifest.readManifest(readTestData("no_manifest.exe"))).isNull();
  }

  @Test
  public void readManifest_notAnExecutable() throws Exception {
    var e =
        org.junit.Assert.assertThrows(
            IllegalArgumentException.class,
            () -> PeManifest.readManifest(readTestData("manifest.xml")));
    assertThat(e).hasMessageThat().contains("not a PE file");
  }

  @Test
  public void checksum_matchesLinker() throws Exception {
    // The test executables were linked with /release, which sets the checksum. This validates
    // the checksum implementation used by assertValid against the linker's.
    for (var name : List.of("with_reloc.exe", "rsrc_last.exe", "pe32.exe")) {
      var headers = new PeHeaders(readTestData(name));
      assertThat(headers.computeChecksum()).isEqualTo(headers.checksum);
    }
  }

  @Test
  public void writeManifest_smallerManifest_rewritesInPlace() throws Exception {
    var original = readTestData("with_reloc.exe");
    var originalHeaders = new PeHeaders(original);

    var rewritten = PeManifest.writeManifest(original, SMALL_MANIFEST);

    assertThat(PeManifest.readManifest(rewritten)).isEqualTo(SMALL_MANIFEST);
    assertThat(rewritten).hasLength(original.length);
    var headers = new PeHeaders(rewritten);
    assertThat(headers.sectionNames()).isEqualTo(originalHeaders.sectionNames());
    for (int i = 0; i < headers.sections.size(); i++) {
      assertThat(headers.sections.get(i).virtualAddress)
          .isEqualTo(originalHeaders.sections.get(i).virtualAddress);
      assertThat(headers.sections.get(i).pointerToRawData)
          .isEqualTo(originalHeaders.sections.get(i).pointerToRawData);
    }
    assertValid(headers, originalHeaders);
    assertOtherResourcesPreserved(rewritten);
  }

  @Test
  public void writeManifest_largerManifest_movesRsrcBehindReloc() throws Exception {
    var original = readTestData("with_reloc.exe");
    var originalHeaders = new PeHeaders(original);
    var manifest = largeManifest();

    var rewritten = PeManifest.writeManifest(original, manifest);

    assertThat(PeManifest.readManifest(rewritten)).isEqualTo(manifest);
    var headers = new PeHeaders(rewritten);
    assertThat(headers.sectionNames()).containsExactly(".text", ".data", ".reloc", ".rsrc").inOrder();
    assertValid(headers, originalHeaders);
    // The base relocations were moved into the space of the old resource section.
    assertThat(headers.section(".reloc").virtualAddress)
        .isEqualTo(originalHeaders.section(".rsrc").virtualAddress);
    assertThat(readSection(rewritten, headers.section(".reloc")))
        .isEqualTo(readSection(original, originalHeaders.section(".reloc")));
    assertOtherResourcesPreserved(rewritten);
  }

  @Test
  public void writeManifest_largerManifest_extendsLastSection() throws Exception {
    var original = readTestData("rsrc_last.exe");
    var originalHeaders = new PeHeaders(original);
    var manifest = largeManifest();

    var rewritten = PeManifest.writeManifest(original, manifest);

    assertThat(PeManifest.readManifest(rewritten)).isEqualTo(manifest);
    var headers = new PeHeaders(rewritten);
    assertThat(headers.sectionNames()).isEqualTo(originalHeaders.sectionNames());
    assertValid(headers, originalHeaders);
    assertThat(headers.section(".rsrc").virtualAddress)
        .isEqualTo(originalHeaders.section(".rsrc").virtualAddress);
    assertThat(headers.section(".rsrc").sizeOfRawData)
        .isGreaterThan(originalHeaders.section(".rsrc").sizeOfRawData);
    assertOtherResourcesPreserved(rewritten);
  }

  @Test
  public void writeManifest_pe32() throws Exception {
    var original = readTestData("pe32.exe");
    var originalHeaders = new PeHeaders(original);
    var manifest = largeManifest();

    var rewritten = PeManifest.writeManifest(original, manifest);

    assertThat(PeManifest.readManifest(rewritten)).isEqualTo(manifest);
    var headers = new PeHeaders(rewritten);
    assertThat(headers.sectionNames()).containsExactly(".text", ".data", ".reloc", ".rsrc").inOrder();
    assertValid(headers, originalHeaders);
    assertOtherResourcesPreserved(rewritten);
  }

  @Test
  public void writeManifest_noManifest_addsManifest() throws Exception {
    var original = readTestData("no_manifest.exe");
    var originalHeaders = new PeHeaders(original);

    var rewritten = PeManifest.writeManifest(original, SMALL_MANIFEST);

    assertThat(PeManifest.readManifest(rewritten)).isEqualTo(SMALL_MANIFEST);
    assertValid(new PeHeaders(rewritten), originalHeaders);
    assertOtherResourcesPreserved(rewritten);
  }

  @Test
  public void writeManifest_noResourceSection_addsSection() throws Exception {
    var original = readTestData("no_rsrc.exe");
    var originalHeaders = new PeHeaders(original);
    assertThat(originalHeaders.sectionNames()).doesNotContain(".rsrc");
    assertThat(PeManifest.readManifest(original)).isNull();
    assertThat(codeViewSignature(original, originalHeaders)).isEqualTo("RSDS");

    var rewritten = PeManifest.writeManifest(original, largeManifest());

    assertThat(PeManifest.readManifest(rewritten)).isEqualTo(largeManifest());
    var headers = new PeHeaders(rewritten);
    assertThat(headers.sectionNames())
        .containsExactly(".text", ".data", ".reloc", ".rsrc")
        .inOrder();
    assertValid(headers, originalHeaders, /* addedSections= */ 1);
    // lld-link leaves no room for another section header, so the headers had to grow and all
    // section data moved. Apart from the file offset of the CodeView record in the debug
    // directory, the section data is unchanged.
    assertThat(headers.sizeOfHeaders).isGreaterThan(originalHeaders.sizeOfHeaders);
    assertThat(codeViewSignature(rewritten, headers)).isEqualTo("RSDS");
    var originalMasked = withoutCodeViewPointer(original, originalHeaders);
    var rewrittenMasked = withoutCodeViewPointer(rewritten, headers);
    for (var section : originalHeaders.sections) {
      var moved = headers.section(section.name());
      assertThat(moved.virtualAddress).isEqualTo(section.virtualAddress);
      assertThat(moved.virtualSize).isEqualTo(section.virtualSize);
      assertThat(readSection(rewrittenMasked, moved))
          .isEqualTo(readSection(originalMasked, section));
    }
  }

  /** Returns the offset of the file offset field of the first debug directory entry. */
  private static int codeViewPointerOffset(PeHeaders headers) {
    return headers.fileOffset(headers.dataDirectory(6)[0]) + 24;
  }

  /** Returns the signature of the CodeView record referenced by the debug directory. */
  private static String codeViewSignature(byte[] image, PeHeaders headers) {
    var pointerToRawData =
        ByteBuffer.wrap(image)
            .order(ByteOrder.LITTLE_ENDIAN)
            .getInt(codeViewPointerOffset(headers));
    return new String(image, pointerToRawData, 4, UTF_8);
  }

  private static byte[] withoutCodeViewPointer(byte[] image, PeHeaders headers) {
    var masked = image.clone();
    java.util.Arrays.fill(masked, codeViewPointerOffset(headers), codeViewPointerOffset(headers) + 4, (byte) 0);
    return masked;
  }

  @Test
  public void writeManifest_stripsSignatureAndPreservesOverlay() throws Exception {
    var original = readTestData("with_reloc.exe");
    var originalHeaders = new PeHeaders(original);
    // Append data after the last section that isn't an Authenticode signature (e.g. Bazel's own
    // self-extracting archive), followed by a fake signature.
    var overlay = "this is an overlay".getBytes(UTF_8);
    var signature = new byte[64];
    java.util.Arrays.fill(signature, (byte) 0x5A);
    var signed = new byte[original.length + overlay.length + signature.length];
    System.arraycopy(original, 0, signed, 0, original.length);
    System.arraycopy(overlay, 0, signed, original.length, overlay.length);
    System.arraycopy(signature, 0, signed, original.length + overlay.length, signature.length);
    var buf = ByteBuffer.wrap(signed).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(originalHeaders.dataDirectoryOffset + 8 * 4, original.length + overlay.length);
    buf.putInt(originalHeaders.dataDirectoryOffset + 8 * 4 + 4, signature.length);

    var rewritten = PeManifest.writeManifest(signed, largeManifest());

    var headers = new PeHeaders(rewritten);
    assertValid(headers, originalHeaders);
    assertThat(headers.dataDirectory(4)).isEqualTo(new int[] {0, 0});
    var last = headers.sections.get(headers.sections.size() - 1);
    var end = last.pointerToRawData + last.sizeOfRawData;
    assertThat(java.util.Arrays.copyOfRange(rewritten, end, rewritten.length)).isEqualTo(overlay);
  }

  private static byte[] largeManifest() {
    return ("<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\" manifestVersion=\"1.0\"><!-- "
            + "x".repeat(20000)
            + " --></assembly>")
        .getBytes(UTF_8);
  }

  private static void assertValid(PeHeaders headers, PeHeaders original) {
    assertValid(headers, original, /* addedSections= */ 0);
  }

  /** Asserts invariants that the Windows loader relies on. */
  private static void assertValid(PeHeaders headers, PeHeaders original, int addedSections) {
    assertThat(headers.sections).hasSize(original.sections.size() + addedSections);
    assertThat(headers.sectionTableEnd).isAtMost(headers.sizeOfHeaders);
    var previousEnd = 0;
    for (var section : headers.sections) {
      assertThat(section.pointerToRawData).isAtLeast(headers.sizeOfHeaders);
      assertThat(section.virtualAddress).isAtLeast(previousEnd);
      assertThat(section.virtualAddress % headers.sectionAlignment).isEqualTo(0);
      assertThat(section.pointerToRawData % headers.fileAlignment).isEqualTo(0);
      assertThat(section.sizeOfRawData % headers.fileAlignment).isEqualTo(0);
      assertThat(section.pointerToRawData + section.sizeOfRawData).isAtMost(headers.fileSize);
      previousEnd = section.virtualAddress + section.virtualSize;
    }
    assertThat(headers.sizeOfImage).isEqualTo(alignUp(previousEnd, headers.sectionAlignment));
    var rsrc = headers.section(".rsrc");
    assertThat(headers.dataDirectory(2)).isEqualTo(new int[] {rsrc.virtualAddress, rsrc.virtualSize});
    if (headers.sectionNames().contains(".reloc")) {
      var reloc = headers.section(".reloc");
      assertThat(headers.dataDirectory(5))
          .isEqualTo(new int[] {reloc.virtualAddress, original.dataDirectory(5)[1]});
    }
    // Sections that weren't moved keep their addresses and only shift in the file if the headers
    // grew.
    var shift = headers.sizeOfHeaders - original.sizeOfHeaders;
    for (var name : List.of(".text", ".data")) {
      assertThat(headers.section(name).virtualAddress)
          .isEqualTo(original.section(name).virtualAddress);
      assertThat(headers.section(name).pointerToRawData)
          .isEqualTo(original.section(name).pointerToRawData + shift);
    }
    assertThat(headers.checksum).isEqualTo(headers.computeChecksum());
  }

  /**
   * Asserts that the icon and string table resources survived the rewrite by checking that their
   * raw data is still present in the image.
   */
  private static void assertOtherResourcesPreserved(byte[] image) throws IOException {
    var ico = Files.readAllBytes(runfile("src/main/cpp/bazel.ico"));
    var icoBuf = ByteBuffer.wrap(ico).order(ByteOrder.LITTLE_ENDIAN);
    int count = Short.toUnsignedInt(icoBuf.getShort(4));
    assertThat(count).isEqualTo(2);
    for (int i = 0; i < count; i++) {
      int size = icoBuf.getInt(6 + 16 * i + 8);
      int offset = icoBuf.getInt(6 + 16 * i + 12);
      assertThat(indexOf(image, java.util.Arrays.copyOfRange(ico, offset, offset + size)))
          .isNotEqualTo(-1);
    }
    // The string table entry "hello" is stored as a length-prefixed UTF-16 string.
    assertThat(indexOf(image, "\u0005hello".getBytes(java.nio.charset.StandardCharsets.UTF_16LE)))
        .isNotEqualTo(-1);
  }

  private static int indexOf(byte[] haystack, byte[] needle) {
    outer:
    for (int i = 0; i + needle.length <= haystack.length; i++) {
      for (int j = 0; j < needle.length; j++) {
        if (haystack[i + j] != needle[j]) {
          continue outer;
        }
      }
      return i;
    }
    return -1;
  }

  private static int alignUp(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
  }

  private static byte[] readSection(byte[] image, SectionHeader section) {
    return java.util.Arrays.copyOfRange(
        image, section.pointerToRawData, section.pointerToRawData + section.sizeOfRawData);
  }

  private static Path runfile(String path) {
    return Path.of(System.getenv("TEST_SRCDIR"), System.getenv("TEST_WORKSPACE"), path);
  }

  private static byte[] readTestData(String name) throws IOException {
    return Files.readAllBytes(runfile(TESTDATA + name));
  }

  private record SectionHeader(
      String name, int virtualSize, int virtualAddress, int sizeOfRawData, int pointerToRawData) {}

  /** An independent, minimal parser for the PE headers relevant to the test. */
  private static final class PeHeaders {
    final int fileSize;
    final int optionalHeaderOffset;
    final int sectionAlignment;
    final int fileAlignment;
    final int sizeOfImage;
    final int sizeOfHeaders;
    final int checksum;
    final int dataDirectoryOffset;
    final int sectionTableEnd;
    final List<SectionHeader> sections = new ArrayList<>();
    private final ByteBuffer buf;

    PeHeaders(byte[] image) {
      fileSize = image.length;
      buf = ByteBuffer.wrap(image).order(ByteOrder.LITTLE_ENDIAN);
      int peOffset = buf.getInt(0x3C);
      assertThat(buf.getInt(peOffset)).isEqualTo(0x00004550);
      int numberOfSections = Short.toUnsignedInt(buf.getShort(peOffset + 6));
      int sizeOfOptionalHeader = Short.toUnsignedInt(buf.getShort(peOffset + 20));
      optionalHeaderOffset = peOffset + 24;
      boolean pe32Plus = Short.toUnsignedInt(buf.getShort(optionalHeaderOffset)) == 0x20B;
      sectionAlignment = buf.getInt(optionalHeaderOffset + 32);
      fileAlignment = buf.getInt(optionalHeaderOffset + 36);
      sizeOfImage = buf.getInt(optionalHeaderOffset + 56);
      sizeOfHeaders = buf.getInt(optionalHeaderOffset + 60);
      checksum = buf.getInt(optionalHeaderOffset + 64);
      dataDirectoryOffset = optionalHeaderOffset + (pe32Plus ? 112 : 96);
      int sectionTableOffset = optionalHeaderOffset + sizeOfOptionalHeader;
      sectionTableEnd = sectionTableOffset + 40 * numberOfSections;
      for (int i = 0; i < numberOfSections; i++) {
        int offset = sectionTableOffset + 40 * i;
        byte[] name = new byte[8];
        buf.get(offset, name);
        sections.add(
            new SectionHeader(
                new String(name, UTF_8).trim().replace("\0", ""),
                buf.getInt(offset + 8),
                buf.getInt(offset + 12),
                buf.getInt(offset + 16),
                buf.getInt(offset + 20)));
      }
    }

    List<String> sectionNames() {
      return sections.stream().map(SectionHeader::name).toList();
    }

    SectionHeader section(String name) {
      return sections.stream().filter(s -> s.name().equals(name)).findFirst().orElseThrow();
    }

    int fileOffset(int rva) {
      var section =
          sections.stream()
              .filter(s -> rva >= s.virtualAddress && rva < s.virtualAddress + s.virtualSize)
              .findFirst()
              .orElseThrow();
      return rva - section.virtualAddress + section.pointerToRawData;
    }

    int[] dataDirectory(int index) {
      return new int[] {
        buf.getInt(dataDirectoryOffset + 8 * index), buf.getInt(dataDirectoryOffset + 8 * index + 4)
      };
    }

    /** The PE checksum algorithm as documented for CheckSumMappedFile. */
    int computeChecksum() {
      int checksumOffset = optionalHeaderOffset + 64;
      long sum = 0;
      for (int i = 0; i + 1 < fileSize; i += 2) {
        if (i == checksumOffset || i == checksumOffset + 2) {
          continue;
        }
        sum += Short.toUnsignedInt(buf.getShort(i));
        sum = (sum & 0xFFFF) + (sum >>> 16);
      }
      if (fileSize % 2 == 1) {
        sum += Byte.toUnsignedInt(buf.get(fileSize - 1));
        sum = (sum & 0xFFFF) + (sum >>> 16);
      }
      sum = (sum & 0xFFFF) + (sum >>> 16);
      return (int) (sum + fileSize);
    }
  }
}
