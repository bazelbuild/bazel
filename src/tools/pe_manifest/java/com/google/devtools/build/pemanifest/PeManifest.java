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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Reads or replaces the application manifest embedded in a Windows executable.
 *
 * <p>The application manifest (also known as the side-by-side or fusion manifest) is stored as
 * the {@code RT_MANIFEST} resource with ID 1 in the {@code .rsrc} section of a PE file. This tool
 * only depends on the PE file format and thus runs on any platform, unlike the {@code
 * UpdateResource} Windows API.
 *
 * <p>Usage:
 *
 * <pre>
 *   pe_manifest read &lt;exe&gt;                    Prints the manifest to stdout.
 *   pe_manifest write &lt;exe&gt; [&lt;manifest&gt;]     Replaces the manifest with the given file
 *                                             (or stdin) and updates the executable in place.
 * </pre>
 *
 * @see <a href="https://learn.microsoft.com/en-us/windows/win32/sbscs/application-manifests">Application Manifests</a>
 * @see <a href="https://learn.microsoft.com/en-us/windows/win32/sbscs/using-side-by-side-assemblies-as-a-resource">Using Side-by-side Assemblies as a Resource</a>
 * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format">PE Format</a>
 */
public final class PeManifest {

  // https://learn.microsoft.com/en-us/windows/win32/menurc/resource-types
  private static final int RT_MANIFEST = 24;
  // https://learn.microsoft.com/en-us/windows/win32/sbscs/using-side-by-side-assemblies-as-a-resource
  private static final int CREATEPROCESS_MANIFEST_RESOURCE_ID = 1;

  private PeManifest() {}

  public static void main(String[] args) throws IOException {
    if (args.length == 2 && args[0].equals("read")) {
      byte[] manifest = readManifest(Files.readAllBytes(Path.of(args[1])));
      if (manifest == null) {
        System.err.println("Error: " + args[1] + " does not contain an application manifest");
        System.exit(1);
      }
      System.out.write(manifest);
      System.out.flush();
    } else if ((args.length == 2 || args.length == 3) && args[0].equals("write")) {
      Path exe = Path.of(args[1]);
      byte[] manifest = args.length == 3 ? Files.readAllBytes(Path.of(args[2])) : readAll(System.in);
      Files.write(exe, writeManifest(Files.readAllBytes(exe), manifest));
    } else {
      PrintStream err = System.err;
      err.println("Usage:");
      err.println("  pe_manifest read <exe>");
      err.println("  pe_manifest write <exe> [<manifest file, defaults to stdin>]");
      System.exit(2);
    }
  }

  /** Returns the application manifest of the given PE image, or null if it doesn't have one. */
  public static byte[] readManifest(byte[] image) {
    PeImage pe = new PeImage(image);
    ResourceSection resources = pe.readResources();
    if (resources == null) {
      return null;
    }
    ResourceDirectory manifests = resources.root.findSubdirectory(RT_MANIFEST);
    if (manifests == null) {
      return null;
    }
    ResourceDirectory languages = manifests.findSubdirectory(CREATEPROCESS_MANIFEST_RESOURCE_ID);
    if (languages == null || languages.entries.isEmpty()) {
      return null;
    }
    DataEntry data = languages.entries.get(0).data;
    return data == null ? null : data.data;
  }

  /**
   * Returns a copy of the given PE image whose application manifest has been replaced with the
   * given one, adding one if the image doesn't have a manifest or even a resource section yet.
   */
  public static byte[] writeManifest(byte[] image, byte[] manifest) {
    PeImage pe = new PeImage(image);
    ResourceSection resources = pe.readResources();
    ResourceDirectory root = resources == null ? new ResourceDirectory() : resources.root;
    ResourceDirectory manifests = root.getOrCreateSubdirectory(RT_MANIFEST);
    ResourceDirectory languages =
        manifests.getOrCreateSubdirectory(CREATEPROCESS_MANIFEST_RESOURCE_ID);
    if (languages.entries.isEmpty()) {
      // Manifests are language neutral.
      languages.entries.add(Entry.forData(0, new DataEntry(manifest, 0)));
    } else {
      for (Entry entry : languages.entries) {
        if (entry.data != null) {
          entry.data = new DataEntry(manifest, entry.data.codePage);
        }
      }
    }
    return resources == null ? pe.addResources(root) : pe.replaceResources(resources);
  }

  private static byte[] readAll(InputStream in) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    in.transferTo(out);
    return out.toByteArray();
  }

  private static int alignUp(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
  }

  /**
   * A section header of a PE image.
   *
   * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#section-table-section-headers">Section Table (Section Headers)</a>
   */
  private static final class Section {
    static final int HEADER_SIZE = 40;

    final String name;
    int virtualSize;
    int virtualAddress;
    int sizeOfRawData;
    int pointerToRawData;
    final byte[] restOfHeader;

    Section(ByteBuffer buf, int offset) {
      byte[] nameBytes = new byte[8];
      buf.get(offset, nameBytes);
      int nameLength = 0;
      while (nameLength < 8 && nameBytes[nameLength] != 0) {
        nameLength++;
      }
      name = new String(nameBytes, 0, nameLength, StandardCharsets.UTF_8);
      virtualSize = buf.getInt(offset + 8);
      virtualAddress = buf.getInt(offset + 12);
      sizeOfRawData = buf.getInt(offset + 16);
      pointerToRawData = buf.getInt(offset + 20);
      restOfHeader = new byte[16];
      buf.get(offset + 24, restOfHeader);
    }

    /**
     * Creates the header of a new section whose remaining fields are all zero.
     *
     * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#section-flags">Section Flags</a>
     */
    Section(
        String name,
        int virtualSize,
        int virtualAddress,
        int sizeOfRawData,
        int pointerToRawData,
        int characteristics) {
      this.name = name;
      this.virtualSize = virtualSize;
      this.virtualAddress = virtualAddress;
      this.sizeOfRawData = sizeOfRawData;
      this.pointerToRawData = pointerToRawData;
      restOfHeader = new byte[16];
      ByteBuffer.wrap(restOfHeader).order(ByteOrder.LITTLE_ENDIAN).putInt(12, characteristics);
    }

    void write(ByteBuffer buf, int offset) {
      byte[] nameBytes = Arrays.copyOf(name.getBytes(StandardCharsets.UTF_8), 8);
      buf.put(offset, nameBytes);
      buf.putInt(offset + 8, virtualSize);
      buf.putInt(offset + 12, virtualAddress);
      buf.putInt(offset + 16, sizeOfRawData);
      buf.putInt(offset + 20, pointerToRawData);
      buf.put(offset + 24, restOfHeader);
    }

    boolean containsRva(int rva) {
      return Integer.compareUnsigned(rva, virtualAddress) >= 0
          && Integer.compareUnsigned(rva, virtualAddress + Math.max(virtualSize, sizeOfRawData))
              < 0;
    }

    int rvaToFileOffset(int rva) {
      return rva - virtualAddress + pointerToRawData;
    }
  }

  /**
   * The parts of a PE image relevant for rewriting its resource section.
   *
   * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format">PE Format</a>
   */
  private static final class PeImage {
    // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#optional-header-data-directories-image-only
    private static final int DATA_DIRECTORY_RESOURCE = 2;
    private static final int DATA_DIRECTORY_SECURITY = 4;
    private static final int DATA_DIRECTORY_DEBUG = 6;
    // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#debug-directory-image-only
    private static final int DEBUG_ENTRY_SIZE = 28;
    // IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
    // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#section-flags
    private static final int RSRC_CHARACTERISTICS = 0x40000040;

    final byte[] image;
    final ByteBuffer buf;
    final int coffHeaderOffset;
    final int optionalHeaderOffset;
    final int dataDirectoryOffset;
    final int numberOfRvaAndSizes;
    final int sectionTableOffset;
    final int sectionAlignment;
    final int fileAlignment;
    final List<Section> sections = new ArrayList<>();

    PeImage(byte[] image) {
      this.image = image;
      this.buf = ByteBuffer.wrap(image).order(ByteOrder.LITTLE_ENDIAN);
      // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#ms-dos-stub-image-only
      if (image.length < 0x40 || buf.getShort(0) != 0x5A4D) {
        throw new IllegalArgumentException("not a PE file: missing MZ signature");
      }
      int peOffset = buf.getInt(0x3C);
      if (peOffset < 0 || peOffset + 24 > image.length || buf.getInt(peOffset) != 0x00004550) {
        throw new IllegalArgumentException("not a PE file: missing PE signature");
      }
      // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#coff-file-header-object-and-image
      coffHeaderOffset = peOffset + 4;
      int numberOfSections = Short.toUnsignedInt(buf.getShort(coffHeaderOffset + 2));
      int sizeOfOptionalHeader = Short.toUnsignedInt(buf.getShort(coffHeaderOffset + 16));
      // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#optional-header-standard-fields-image-only
      optionalHeaderOffset = coffHeaderOffset + 20;
      int magic = Short.toUnsignedInt(buf.getShort(optionalHeaderOffset));
      boolean pe32Plus =
          switch (magic) {
            case 0x10B -> false;
            case 0x20B -> true;
            default -> throw new IllegalArgumentException("unsupported optional header magic");
          };
      // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#optional-header-windows-specific-fields-image-only
      sectionAlignment = buf.getInt(optionalHeaderOffset + 32);
      fileAlignment = buf.getInt(optionalHeaderOffset + 36);
      numberOfRvaAndSizes = buf.getInt(optionalHeaderOffset + (pe32Plus ? 108 : 92));
      dataDirectoryOffset = optionalHeaderOffset + (pe32Plus ? 112 : 96);
      sectionTableOffset = optionalHeaderOffset + sizeOfOptionalHeader;
      for (int i = 0; i < numberOfSections; i++) {
        sections.add(new Section(buf, sectionTableOffset + i * Section.HEADER_SIZE));
      }
    }

    private int dataDirectoryRva(int index) {
      return index < numberOfRvaAndSizes ? buf.getInt(dataDirectoryOffset + 8 * index) : 0;
    }

    private int dataDirectorySize(int index) {
      return index < numberOfRvaAndSizes ? buf.getInt(dataDirectoryOffset + 8 * index + 4) : 0;
    }

    private Section sectionContaining(int rva) {
      for (Section section : sections) {
        if (section.containsRva(rva)) {
          return section;
        }
      }
      return null;
    }

    // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#the-rsrc-section
    ResourceSection readResources() {
      int rva = dataDirectoryRva(DATA_DIRECTORY_RESOURCE);
      if (rva == 0) {
        return null;
      }
      Section section = sectionContaining(rva);
      if (section == null) {
        throw new IllegalArgumentException("resource directory is not inside any section");
      }
      if (rva != section.virtualAddress) {
        throw new IllegalArgumentException(
            "unsupported executable: resource directory does not start at the beginning of the "
                + section.name
                + " section");
      }
      return new ResourceSection(section, ResourceDirectory.parse(this, section, 0));
    }

    /** Returns a copy of the image with the given resources written back into their section. */
    byte[] replaceResources(ResourceSection resources) {
      Section rsrc = resources.section;
      int index = sections.indexOf(rsrc);
      // Try to fit the new resource section into the space of the existing one first. The space
      // covers the slack up to the file and section alignment of the existing raw data.
      int newSize = resources.root.serializedSize();
      int nextVirtualAddress =
          index + 1 < sections.size()
              ? sections.get(index + 1).virtualAddress
              : alignUp(rsrc.virtualAddress + rsrc.virtualSize, sectionAlignment);
      if (newSize <= rsrc.sizeOfRawData && rsrc.virtualAddress + newSize <= nextVirtualAddress) {
        byte[] result = image.clone();
        ByteBuffer out = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN);
        byte[] serialized = resources.root.serialize(rsrc.virtualAddress);
        Arrays.fill(result, rsrc.pointerToRawData, rsrc.pointerToRawData + rsrc.sizeOfRawData, (byte) 0);
        System.arraycopy(serialized, 0, result, rsrc.pointerToRawData, serialized.length);
        rsrc.virtualSize = newSize;
        rsrc.write(out, sectionTableOffset + index * Section.HEADER_SIZE);
        out.putInt(dataDirectoryOffset + 8 * DATA_DIRECTORY_RESOURCE + 4, newSize);
        stripSignatureAndUpdateChecksum(out);
        return result;
      }
      // Otherwise, move the resource section to the end of the image, swapping it with a trailing
      // .reloc section, whose entries only reference the pages of other sections.
      // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#the-reloc-section-image-only
      List<Section> following = sections.subList(index + 1, sections.size());
      if (following.size() > 1 || (following.size() == 1 && !following.get(0).name.equals(".reloc"))) {
        throw new IllegalArgumentException(
            "unsupported executable: the new manifest does not fit into the existing resource "
                + "section and the section is followed by sections other than .reloc");
      }
      Section reloc = following.isEmpty() ? null : following.get(0);
      int lastSectionEnd = rsrc.pointerToRawData + rsrc.sizeOfRawData;
      if (reloc != null) {
        lastSectionEnd = Math.max(lastSectionEnd, reloc.pointerToRawData + reloc.sizeOfRawData);
      }
      byte[] overlay = overlayWithoutSignature(lastSectionEnd);

      int oldRsrcVirtualAddress = rsrc.virtualAddress;
      int oldRsrcPointerToRawData = rsrc.pointerToRawData;
      int oldRelocVirtualAddress = reloc == null ? 0 : reloc.virtualAddress;
      byte[] relocData =
          reloc == null
              ? new byte[0]
              : Arrays.copyOfRange(
                  image, reloc.pointerToRawData, reloc.pointerToRawData + reloc.sizeOfRawData);
      int rawOffset = rsrc.pointerToRawData;
      int virtualAddress = rsrc.virtualAddress;
      if (reloc != null) {
        reloc.pointerToRawData = rawOffset;
        reloc.virtualAddress = virtualAddress;
        rawOffset = alignUp(rawOffset + reloc.sizeOfRawData, fileAlignment);
        virtualAddress = alignUp(virtualAddress + reloc.virtualSize, sectionAlignment);
      }
      rsrc.pointerToRawData = rawOffset;
      rsrc.virtualAddress = virtualAddress;
      rsrc.virtualSize = newSize;
      rsrc.sizeOfRawData = alignUp(newSize, fileAlignment);
      byte[] serialized = resources.root.serialize(rsrc.virtualAddress);

      byte[] result =
          Arrays.copyOf(image, rsrc.pointerToRawData + rsrc.sizeOfRawData + overlay.length);
      ByteBuffer out = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN);
      Arrays.fill(result, oldRsrcPointerToRawData, result.length, (byte) 0);
      if (reloc != null) {
        System.arraycopy(relocData, 0, result, reloc.pointerToRawData, relocData.length);
      }
      System.arraycopy(serialized, 0, result, rsrc.pointerToRawData, serialized.length);
      System.arraycopy(
          overlay, 0, result, rsrc.pointerToRawData + rsrc.sizeOfRawData, overlay.length);

      // Sections must be listed in ascending order of their virtual addresses.
      if (reloc != null) {
        sections.set(index, reloc);
        sections.set(index + 1, rsrc);
      }
      for (int i = 0; i < sections.size(); i++) {
        sections.get(i).write(out, sectionTableOffset + i * Section.HEADER_SIZE);
      }
      Section last = sections.get(sections.size() - 1);
      // SizeOfImage
      out.putInt(
          optionalHeaderOffset + 56, alignUp(last.virtualAddress + last.virtualSize, sectionAlignment));
      // Translate all data directories that point into the moved sections. The certificate table
      // holds a file offset instead of an RVA and is cleared below.
      // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#optional-header-data-directories-image-only
      for (int i = 0; i < numberOfRvaAndSizes; i++) {
        if (i == DATA_DIRECTORY_SECURITY) {
          continue;
        }
        int rva = dataDirectoryRva(i);
        if (rva == 0) {
          continue;
        }
        if (reloc != null
            && Integer.compareUnsigned(rva, oldRelocVirtualAddress) >= 0
            && Integer.compareUnsigned(rva, oldRelocVirtualAddress + reloc.virtualSize) < 0) {
          out.putInt(dataDirectoryOffset + 8 * i, rva - oldRelocVirtualAddress + reloc.virtualAddress);
        } else if (Integer.compareUnsigned(rva, oldRsrcVirtualAddress) >= 0
            && Integer.compareUnsigned(rva, oldRelocVirtualAddress == 0 ? rsrc.virtualAddress : oldRelocVirtualAddress) < 0) {
          out.putInt(dataDirectoryOffset + 8 * i, rva - oldRsrcVirtualAddress + rsrc.virtualAddress);
        }
      }
      out.putInt(dataDirectoryOffset + 8 * DATA_DIRECTORY_RESOURCE + 4, newSize);
      stripSignatureAndUpdateChecksum(out);
      return result;
    }

    /**
     * Returns a copy of the image with the given resources in a new {@code .rsrc} section
     * appended after the last section, as needed e.g. for GraalVM native images. If the section
     * table has no room for another header, the headers grow by a multiple of the file alignment
     * and all section data moves accordingly.
     *
     * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#section-table-section-headers">Section Table (Section Headers)</a>
     * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#optional-header-windows-specific-fields-image-only">Optional Header Windows-Specific Fields</a>
     */
    byte[] addResources(ResourceDirectory root) {
      if (DATA_DIRECTORY_RESOURCE >= numberOfRvaAndSizes) {
        throw new IllegalArgumentException("unsupported executable: no resource table entry");
      }
      int sizeOfHeaders = buf.getInt(optionalHeaderOffset + 60);
      int lastSectionEnd = 0;
      for (Section section : sections) {
        lastSectionEnd = Math.max(lastSectionEnd, section.pointerToRawData + section.sizeOfRawData);
      }
      if (sizeOfHeaders > lastSectionEnd || sizeOfHeaders % fileAlignment != 0) {
        throw new IllegalArgumentException("unsupported executable: invalid SizeOfHeaders");
      }
      int sectionTableEnd = sectionTableOffset + (sections.size() + 1) * Section.HEADER_SIZE;
      int shift =
          sectionTableEnd > sizeOfHeaders
              ? alignUp(sectionTableEnd - sizeOfHeaders, fileAlignment)
              : 0;
      byte[] overlay = overlayWithoutSignature(lastSectionEnd);
      Section last = sections.get(sections.size() - 1);
      int newSize = root.serializedSize();
      Section rsrc =
          new Section(
              ".rsrc",
              newSize,
              alignUp(last.virtualAddress + last.virtualSize, sectionAlignment),
              alignUp(newSize, fileAlignment),
              alignUp(lastSectionEnd, fileAlignment) + shift,
              RSRC_CHARACTERISTICS);
      byte[] serialized = root.serialize(rsrc.virtualAddress);

      byte[] result = new byte[rsrc.pointerToRawData + rsrc.sizeOfRawData + overlay.length];
      System.arraycopy(image, 0, result, 0, sizeOfHeaders);
      System.arraycopy(
          image, sizeOfHeaders, result, sizeOfHeaders + shift, lastSectionEnd - sizeOfHeaders);
      System.arraycopy(serialized, 0, result, rsrc.pointerToRawData, serialized.length);
      System.arraycopy(
          overlay, 0, result, rsrc.pointerToRawData + rsrc.sizeOfRawData, overlay.length);

      ByteBuffer out = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN);
      if (shift != 0) {
        shiftFileOffsets(out, shift);
        // SizeOfHeaders
        out.putInt(optionalHeaderOffset + 60, sizeOfHeaders + shift);
        for (Section section : sections) {
          if (section.pointerToRawData != 0) {
            section.pointerToRawData += shift;
          }
        }
      }
      sections.add(rsrc);
      // NumberOfSections
      out.putShort(coffHeaderOffset + 2, (short) sections.size());
      for (int i = 0; i < sections.size(); i++) {
        sections.get(i).write(out, sectionTableOffset + i * Section.HEADER_SIZE);
      }
      // SizeOfInitializedData
      out.putInt(
          optionalHeaderOffset + 8, buf.getInt(optionalHeaderOffset + 8) + rsrc.sizeOfRawData);
      // SizeOfImage
      out.putInt(
          optionalHeaderOffset + 56, alignUp(rsrc.virtualAddress + rsrc.virtualSize, sectionAlignment));
      out.putInt(dataDirectoryOffset + 8 * DATA_DIRECTORY_RESOURCE, rsrc.virtualAddress);
      out.putInt(dataDirectoryOffset + 8 * DATA_DIRECTORY_RESOURCE + 4, newSize);
      stripSignatureAndUpdateChecksum(out);
      return result;
    }

    /**
     * Updates the file offsets stored outside of the section table after all section data has
     * moved by the given amount. Must be called before the section table is updated.
     *
     * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#coff-file-header-object-and-image">COFF File Header</a>
     * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#debug-directory-image-only">Debug Directory</a>
     */
    private void shiftFileOffsets(ByteBuffer out, int shift) {
      // PointerToSymbolTable
      int symbolTable = buf.getInt(coffHeaderOffset + 8);
      if (symbolTable != 0) {
        out.putInt(coffHeaderOffset + 8, symbolTable + shift);
      }
      int debugRva = dataDirectoryRva(DATA_DIRECTORY_DEBUG);
      Section debugSection = debugRva == 0 ? null : sectionContaining(debugRva);
      if (debugSection == null) {
        return;
      }
      int debugOffset = debugSection.rvaToFileOffset(debugRva);
      int debugSize = dataDirectorySize(DATA_DIRECTORY_DEBUG);
      for (int entry = 0; entry + DEBUG_ENTRY_SIZE <= debugSize; entry += DEBUG_ENTRY_SIZE) {
        // PointerToRawData
        int pointerOffset = debugOffset + entry + 24;
        int pointer = buf.getInt(pointerOffset);
        if (pointer != 0) {
          out.putInt(pointerOffset + shift, pointer + shift);
        }
      }
    }

    /**
     * Returns the data following the last section (an overlay), which is preserved except for the
     * Authenticode signature, which the modification invalidates anyway.
     *
     * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#the-attribute-certificate-table-image-only">The Attribute Certificate Table</a>
     */
    private byte[] overlayWithoutSignature(int lastSectionEnd) {
      int securityOffset = dataDirectoryRva(DATA_DIRECTORY_SECURITY);
      int securitySize = dataDirectorySize(DATA_DIRECTORY_SECURITY);
      if (securityOffset < lastSectionEnd || securityOffset + securitySize > image.length) {
        return Arrays.copyOfRange(image, lastSectionEnd, image.length);
      }
      byte[] overlay = new byte[image.length - lastSectionEnd - securitySize];
      System.arraycopy(image, lastSectionEnd, overlay, 0, securityOffset - lastSectionEnd);
      System.arraycopy(
          image,
          securityOffset + securitySize,
          overlay,
          securityOffset - lastSectionEnd,
          image.length - securityOffset - securitySize);
      return overlay;
    }

    private void stripSignatureAndUpdateChecksum(ByteBuffer out) {
      if (DATA_DIRECTORY_SECURITY < numberOfRvaAndSizes) {
        out.putInt(dataDirectoryOffset + 8 * DATA_DIRECTORY_SECURITY, 0);
        out.putInt(dataDirectoryOffset + 8 * DATA_DIRECTORY_SECURITY + 4, 0);
      }
      int checksumOffset = optionalHeaderOffset + 64;
      out.putInt(checksumOffset, 0);
      out.putInt(checksumOffset, checksum(out, checksumOffset));
    }

    /**
     * Computes the PE checksum as implemented by {@code CheckSumMappedFile} in imagehlp.dll.
     *
     * @see <a href="https://learn.microsoft.com/en-us/windows/win32/api/imagehlp/nf-imagehlp-checksummappedfile">CheckSumMappedFile</a>
     */
    private static int checksum(ByteBuffer buf, int checksumOffset) {
      long sum = 0;
      int length = buf.limit();
      for (int i = 0; i + 1 < length; i += 2) {
        if (i == checksumOffset || i == checksumOffset + 2) {
          continue;
        }
        sum += Short.toUnsignedInt(buf.getShort(i));
        sum = (sum & 0xFFFF) + (sum >>> 16);
      }
      if (length % 2 == 1) {
        sum += Byte.toUnsignedInt(buf.get(length - 1));
        sum = (sum & 0xFFFF) + (sum >>> 16);
      }
      sum = (sum & 0xFFFF) + (sum >>> 16);
      return (int) (sum + length);
    }
  }

  /** The parsed resource tree of a PE image together with the section it was read from. */
  private static final class ResourceSection {
    final Section section;
    final ResourceDirectory root;

    ResourceSection(Section section, ResourceDirectory root) {
      this.section = section;
      this.root = root;
    }
  }

  /**
   * A resource data entry, i.e. a leaf of the resource tree.
   *
   * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#resource-data-entry">Resource Data Entry</a>
   */
  private static final class DataEntry {
    final byte[] data;
    final int codePage;

    DataEntry(byte[] data, int codePage) {
      this.data = data;
      this.codePage = codePage;
    }
  }

  /**
   * An entry of a resource directory, identified by either a name or an integer ID.
   *
   * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#resource-directory-entries">Resource Directory Entries</a>
   */
  private static final class Entry {
    final String name;
    final int id;
    ResourceDirectory subdirectory;
    DataEntry data;

    private Entry(String name, int id) {
      this.name = name;
      this.id = id;
    }

    static Entry forSubdirectory(int id, ResourceDirectory subdirectory) {
      Entry entry = new Entry(null, id);
      entry.subdirectory = subdirectory;
      return entry;
    }

    static Entry forData(int id, DataEntry data) {
      Entry entry = new Entry(null, id);
      entry.data = data;
      return entry;
    }
  }

  /**
   * A resource directory, i.e. an inner node of the three-level resource tree.
   *
   * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#resource-directory-table">Resource Directory Table</a>
   * @see <a href="https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#the-rsrc-section">The .rsrc Section</a>
   */
  private static final class ResourceDirectory {
    private static final int HEADER_SIZE = 16;
    private static final int ENTRY_SIZE = 8;
    private static final int DATA_ENTRY_SIZE = 16;
    private static final int DATA_ALIGNMENT = 8;

    final int characteristics;
    final int timeDateStamp;
    final short majorVersion;
    final short minorVersion;
    final List<Entry> entries = new ArrayList<>();

    private ResourceDirectory(
        int characteristics, int timeDateStamp, short majorVersion, short minorVersion) {
      this.characteristics = characteristics;
      this.timeDateStamp = timeDateStamp;
      this.majorVersion = majorVersion;
      this.minorVersion = minorVersion;
    }

    ResourceDirectory() {
      this(0, 0, (short) 0, (short) 0);
    }

    static ResourceDirectory parse(PeImage pe, Section section, int offset) {
      ByteBuffer buf = pe.buf;
      int base = section.pointerToRawData;
      ResourceDirectory directory =
          new ResourceDirectory(
              buf.getInt(base + offset),
              buf.getInt(base + offset + 4),
              buf.getShort(base + offset + 8),
              buf.getShort(base + offset + 10));
      int numberOfNamedEntries = Short.toUnsignedInt(buf.getShort(base + offset + 12));
      int numberOfIdEntries = Short.toUnsignedInt(buf.getShort(base + offset + 14));
      int entryOffset = offset + HEADER_SIZE;
      for (int i = 0; i < numberOfNamedEntries + numberOfIdEntries; i++) {
        int nameOrId = buf.getInt(base + entryOffset);
        int dataOrSubdirectory = buf.getInt(base + entryOffset + 4);
        Entry entry;
        if ((nameOrId & 0x80000000) != 0) {
          int nameOffset = base + (nameOrId & 0x7FFFFFFF);
          int length = Short.toUnsignedInt(buf.getShort(nameOffset));
          byte[] chars = new byte[2 * length];
          buf.get(nameOffset + 2, chars);
          entry = new Entry(new String(chars, StandardCharsets.UTF_16LE), 0);
        } else {
          entry = new Entry(null, nameOrId);
        }
        if ((dataOrSubdirectory & 0x80000000) != 0) {
          entry.subdirectory = parse(pe, section, dataOrSubdirectory & 0x7FFFFFFF);
        } else {
          int dataEntryOffset = base + dataOrSubdirectory;
          int dataRva = buf.getInt(dataEntryOffset);
          int size = buf.getInt(dataEntryOffset + 4);
          int codePage = buf.getInt(dataEntryOffset + 8);
          Section dataSection = pe.sectionContaining(dataRva);
          if (dataSection == null) {
            throw new IllegalArgumentException("resource data is not inside any section");
          }
          int dataOffset = dataSection.rvaToFileOffset(dataRva);
          entry.data =
              new DataEntry(Arrays.copyOfRange(pe.image, dataOffset, dataOffset + size), codePage);
        }
        directory.entries.add(entry);
        entryOffset += ENTRY_SIZE;
      }
      return directory;
    }

    ResourceDirectory findSubdirectory(int id) {
      for (Entry entry : entries) {
        if (entry.name == null && entry.id == id) {
          return entry.subdirectory;
        }
      }
      return null;
    }

    ResourceDirectory getOrCreateSubdirectory(int id) {
      ResourceDirectory existing = findSubdirectory(id);
      if (existing != null) {
        return existing;
      }
      ResourceDirectory subdirectory = new ResourceDirectory();
      // Entries are sorted with named entries first, followed by ID entries in ascending order.
      // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#resource-directory-table
      int position = entries.size();
      for (int i = 0; i < entries.size(); i++) {
        Entry entry = entries.get(i);
        if (entry.name == null && Integer.compareUnsigned(entry.id, id) > 0) {
          position = i;
          break;
        }
      }
      entries.add(position, Entry.forSubdirectory(id, subdirectory));
      return subdirectory;
    }

    int serializedSize() {
      return serialize(0).length;
    }

    /**
     * Serializes the resource tree in the standard layout: all directory tables in breadth-first
     * order, followed by the data entries, the string table and finally the resource data.
     *
     * @param virtualAddress the RVA at which the serialized tree will be placed, which the data
     *     entries reference their contents by
     */
    byte[] serialize(int virtualAddress) {
      List<ResourceDirectory> directories = new ArrayList<>();
      List<Entry> leaves = new ArrayList<>();
      List<String> names = new ArrayList<>();
      collect(directories, leaves, names);

      int directoriesSize = 0;
      for (ResourceDirectory directory : directories) {
        directoriesSize += HEADER_SIZE + ENTRY_SIZE * directory.entries.size();
      }
      int dataEntriesOffset = directoriesSize;
      int stringsOffset = dataEntriesOffset + DATA_ENTRY_SIZE * leaves.size();
      int stringsSize = 0;
      for (String name : names) {
        stringsSize += alignUp(2 + 2 * name.length(), 2);
      }
      int dataOffset = alignUp(stringsOffset + stringsSize, DATA_ALIGNMENT);
      int totalSize = dataOffset;
      for (Entry leaf : leaves) {
        totalSize = alignUp(totalSize + leaf.data.data.length, DATA_ALIGNMENT);
      }

      ByteBuffer out = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN);
      // Offsets of all directories and names have to be known before the entries can be written.
      List<Integer> directoryOffsets = new ArrayList<>();
      int offset = 0;
      for (ResourceDirectory directory : directories) {
        directoryOffsets.add(offset);
        offset += HEADER_SIZE + ENTRY_SIZE * directory.entries.size();
      }
      List<Integer> nameOffsets = new ArrayList<>();
      offset = stringsOffset;
      for (String name : names) {
        nameOffsets.add(offset);
        out.putShort(offset, (short) name.length());
        out.put(offset + 2, name.getBytes(StandardCharsets.UTF_16LE));
        offset += alignUp(2 + 2 * name.length(), 2);
      }
      int dataEntryOffset = dataEntriesOffset;
      offset = dataOffset;
      for (Entry leaf : leaves) {
        out.putInt(dataEntryOffset, virtualAddress + offset);
        out.putInt(dataEntryOffset + 4, leaf.data.data.length);
        out.putInt(dataEntryOffset + 8, leaf.data.codePage);
        out.putInt(dataEntryOffset + 12, 0);
        out.put(offset, leaf.data.data);
        dataEntryOffset += DATA_ENTRY_SIZE;
        offset = alignUp(offset + leaf.data.data.length, DATA_ALIGNMENT);
      }
      int leafIndex = 0;
      int nameIndex = 0;
      for (int i = 0; i < directories.size(); i++) {
        ResourceDirectory directory = directories.get(i);
        offset = directoryOffsets.get(i);
        out.putInt(offset, directory.characteristics);
        out.putInt(offset + 4, directory.timeDateStamp);
        out.putShort(offset + 8, directory.majorVersion);
        out.putShort(offset + 10, directory.minorVersion);
        int numberOfNamedEntries = 0;
        for (Entry entry : directory.entries) {
          if (entry.name != null) {
            numberOfNamedEntries++;
          }
        }
        out.putShort(offset + 12, (short) numberOfNamedEntries);
        out.putShort(offset + 14, (short) (directory.entries.size() - numberOfNamedEntries));
        offset += HEADER_SIZE;
        for (Entry entry : directory.entries) {
          if (entry.name != null) {
            out.putInt(offset, 0x80000000 | nameOffsets.get(nameIndex++));
          } else {
            out.putInt(offset, entry.id);
          }
          if (entry.subdirectory != null) {
            out.putInt(offset + 4, 0x80000000 | directoryOffsets.get(directories.indexOf(entry.subdirectory)));
          } else {
            out.putInt(offset + 4, dataEntriesOffset + DATA_ENTRY_SIZE * leafIndex++);
          }
          offset += ENTRY_SIZE;
        }
      }
      return out.array();
    }

    /**
     * Collects all directories in breadth-first order as well as all leaves and names in the
     * order in which they are referenced by the directory entries.
     */
    private void collect(List<ResourceDirectory> directories, List<Entry> leaves, List<String> names) {
      directories.add(this);
      for (int i = 0; i < directories.size(); i++) {
        for (Entry entry : directories.get(i).entries) {
          if (entry.name != null) {
            names.add(entry.name);
          }
          if (entry.subdirectory != null) {
            directories.add(entry.subdirectory);
          } else {
            leaves.add(entry);
          }
        }
      }
    }
  }
}
