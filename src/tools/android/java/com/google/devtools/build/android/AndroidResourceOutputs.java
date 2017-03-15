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
package com.google.devtools.build.android;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.collect.Ordering;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Objects;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/** Collects all the functionationality for an action to create the final output artifacts. */
public class AndroidResourceOutputs {

  /** A FileVisitor that will add all R class files to be stored in a zip archive. */
  static final class ClassJarBuildingVisitor extends ZipBuilderVisitor {

    ClassJarBuildingVisitor(ZipOutputStream zip, Path root) {
      super(zip, root, null);
    }

    private byte[] manifestContent() throws IOException {
      Manifest manifest = new Manifest();
      Attributes attributes = manifest.getMainAttributes();
      attributes.put(Attributes.Name.MANIFEST_VERSION, "1.0");
      Attributes.Name createdBy = new Attributes.Name("Created-By");
      if (attributes.getValue(createdBy) == null) {
        attributes.put(createdBy, "bazel");
      }
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      manifest.write(out);
      return out.toByteArray();
    }

    @Override
    protected void writeFileEntry(Path file) throws IOException {
      Path filename = file.getFileName();
      String name = filename.toString();
      if (name.endsWith(".class")) {
        byte[] content = Files.readAllBytes(file);
        addEntry(file, content);
      }
    }

    void writeManifestContent() throws IOException {
      addEntry(root.resolve(JarFile.MANIFEST_NAME), manifestContent());
    }
  }

  /** A FileVisitor that will add all R.java files to be stored in a zip archive. */
  static final class SymbolFileSrcJarBuildingVisitor extends ZipBuilderVisitor {

    static final Pattern ID_PATTERN =
        Pattern.compile("public static int ([\\w\\.]+)=0x[0-9A-fa-f]+;");
    static final Pattern INNER_CLASS =
        Pattern.compile("public static class ([a-z_]*) \\{(.*?)\\}", Pattern.DOTALL);
    static final Pattern PACKAGE_PATTERN =
        Pattern.compile("\\s*package ([a-zA-Z_$][a-zA-Z\\d_$]*(?:\\.[a-zA-Z_$][a-zA-Z\\d_$]*)*)");

    private final boolean staticIds;

    private SymbolFileSrcJarBuildingVisitor(ZipOutputStream zip, Path root, boolean staticIds) {
      super(zip, root, null);
      this.staticIds = staticIds;
    }

    private String replaceIdsWithStaticIds(String contents) {
      Matcher packageMatcher = PACKAGE_PATTERN.matcher(contents);
      if (!packageMatcher.find()) {
        return contents;
      }
      String pkg = packageMatcher.group(1);
      StringBuffer out = new StringBuffer();
      Matcher innerClassMatcher = INNER_CLASS.matcher(contents);
      while (innerClassMatcher.find()) {
        String resourceType = innerClassMatcher.group(1);
        Matcher idMatcher = ID_PATTERN.matcher(innerClassMatcher.group(2));
        StringBuffer resourceIds = new StringBuffer();
        while (idMatcher.find()) {
          String javaId = idMatcher.group(1);
          idMatcher.appendReplacement(
              resourceIds,
              String.format(
                  "public static int %s=0x%08X;", javaId, Objects.hash(pkg, resourceType, javaId)));
        }
        idMatcher.appendTail(resourceIds);
        innerClassMatcher.appendReplacement(
            out,
            String.format("public static class %s {%s}", resourceType, resourceIds.toString()));
      }
      innerClassMatcher.appendTail(out);
      return out.toString();
    }

    @Override
    protected void writeFileEntry(Path file) throws IOException {
      if (file.getFileName().endsWith("R.java")) {
        byte[] content = Files.readAllBytes(file);
        if (staticIds) {
          content =
              replaceIdsWithStaticIds(UTF_8.decode(ByteBuffer.wrap(content)).toString())
                  .getBytes(UTF_8);
        }
        addEntry(file, content);
      }
    }
  }

  /** A FileVisitor that will add all files to be stored in a zip archive. */
  static class ZipBuilderVisitor extends SimpleFileVisitor<Path> {

    // ZIP timestamps have a resolution of 2 seconds.
    // see http://www.info-zip.org/FAQ.html#limits
    private static final long MINIMUM_TIMESTAMP_INCREMENT = 2000L;
    // The earliest date representable in a zip file, 1-1-1980 (the DOS epoch).
    private static final long ZIP_EPOCH = 315561600000L;

    private final String directoryPrefix;
    private final Collection<Path> paths = new ArrayList<>();
    protected final Path root;
    private int storageMethod = ZipEntry.STORED;
    private final ZipOutputStream zip;

    ZipBuilderVisitor(ZipOutputStream zip, Path root, String directory) {
      this.zip = zip;
      this.root = root;
      this.directoryPrefix = directory;
    }

    protected void addEntry(Path file, byte[] content) throws IOException {
      String prefix = directoryPrefix != null ? (directoryPrefix + "/") : "";
      String relativeName = root.relativize(file).toString();
      ZipEntry entry = new ZipEntry(prefix + relativeName);
      entry.setMethod(storageMethod);
      entry.setTime(normalizeTime(relativeName));
      entry.setSize(content.length);
      CRC32 crc32 = new CRC32();
      crc32.update(content);
      entry.setCrc(crc32.getValue());

      zip.putNextEntry(entry);
      zip.write(content);
      zip.closeEntry();
    }

    /**
     * Normalize timestamps for deterministic builds. Stamp .class files to be a bit newer than
     * .java files. See: {@link
     * com.google.devtools.build.buildjar.jarhelper.JarHelper#normalizedTimestamp(String)}
     */
    protected long normalizeTime(String filename) {
      if (filename.endsWith(".class")) {
        return ZIP_EPOCH + MINIMUM_TIMESTAMP_INCREMENT;
      } else {
        return ZIP_EPOCH;
      }
    }

    public void setCompress(boolean compress) {
      storageMethod = compress ? ZipEntry.DEFLATED : ZipEntry.STORED;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      paths.add(file);
      return FileVisitResult.CONTINUE;
    }

    /**
     * Iterate through collected file paths in a deterministic order and write to the zip.
     *
     * @throws IOException if there is an error reading from the source or writing to the zip.
     */
    void writeEntries() throws IOException {
      for (Path path : Ordering.natural().immutableSortedCopy(paths)) {
        writeFileEntry(path);
      }
    }

    protected void writeFileEntry(Path file) throws IOException {
      byte[] content = Files.readAllBytes(file);
      addEntry(file, content);
    }
  }

  static final Pattern HEX_REGEX = Pattern.compile("0x[0-9A-Fa-f]{8}");

  /**
   * Copies the AndroidManifest.xml to the specified output location.
   *
   * @param androidData The MergedAndroidData which contains the manifest to be written to
   *     manifestOut.
   * @param manifestOut The Path to write the AndroidManifest.xml.
   */
  public static void copyManifestToOutput(MergedAndroidData androidData, Path manifestOut) {
    try {
      Files.createDirectories(manifestOut.getParent());
      Files.copy(androidData.getManifest(), manifestOut);
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(manifestOut, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Copies the R.txt to the expected place.
   *
   * @param generatedSourceRoot The path to the generated R.txt.
   * @param rOutput The Path to write the R.txt.
   * @param staticIds Boolean that indicates if the ids should be set to 0x1 for caching purposes.
   */
  public static void copyRToOutput(Path generatedSourceRoot, Path rOutput, boolean staticIds) {
    try {
      Files.createDirectories(rOutput.getParent());
      final Path source = generatedSourceRoot.resolve("R.txt");
      if (Files.exists(source)) {
        if (staticIds) {
          String contents =
              HEX_REGEX
                  .matcher(Joiner.on("\n").join(Files.readAllLines(source, UTF_8)))
                  .replaceAll("0x1");
          Files.write(rOutput, contents.getBytes(UTF_8));
        } else {
          Files.copy(source, rOutput);
        }
      } else {
        // The R.txt wasn't generated, create one for future inheritance, as Bazel always requires
        // outputs. This state occurs when there are no resource directories.
        Files.createFile(rOutput);
      }
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(rOutput, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /** Creates a zip archive from all found R.class (and inner class) files. */
  public static void createClassJar(Path generatedClassesRoot, Path classJar) {
    try {
      Files.createDirectories(classJar.getParent());
      try (final ZipOutputStream zip =
          new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(classJar)))) {
        ClassJarBuildingVisitor visitor =
            new ClassJarBuildingVisitor(zip, generatedClassesRoot);
        Files.walkFileTree(generatedClassesRoot, visitor);
        visitor.writeEntries();
        visitor.writeManifestContent();
      }
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(classJar, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Creates a zip file containing the provided android resources and assets.
   *
   * @param resourcesRoot The root containing android resources to be written.
   * @param assetsRoot The root containing android assets to be written.
   * @param output The path to write the zip file
   * @param compress Whether or not to compress the content
   * @throws IOException
   */
  public static void createResourcesZip(
      Path resourcesRoot, Path assetsRoot, Path output, boolean compress) throws IOException {
    try (ZipOutputStream zout =
        new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(output)))) {
      if (Files.exists(resourcesRoot)) {
        ZipBuilderVisitor visitor =
            new ZipBuilderVisitor(zout, resourcesRoot, "res");
        visitor.setCompress(compress);
        Files.walkFileTree(resourcesRoot, visitor);
        visitor.writeEntries();
      }
      if (Files.exists(assetsRoot)) {
        ZipBuilderVisitor visitor =
            new ZipBuilderVisitor(zout, assetsRoot, "assets");
        visitor.setCompress(compress);
        Files.walkFileTree(assetsRoot, visitor);
        visitor.writeEntries();
      }
    }
  }

  /** Creates a zip archive from all found R.java files. */
  public static void createSrcJar(Path generatedSourcesRoot, Path srcJar, boolean staticIds) {
    try {
      Files.createDirectories(srcJar.getParent());
      try (final ZipOutputStream zip =
          new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(srcJar)))) {
        SymbolFileSrcJarBuildingVisitor visitor =
            new SymbolFileSrcJarBuildingVisitor(
                zip, generatedSourcesRoot, staticIds);
        Files.walkFileTree(generatedSourcesRoot, visitor);
        visitor.writeEntries();
      }
      // Set to the epoch for caching purposes.
      Files.setLastModifiedTime(srcJar, FileTime.fromMillis(0L));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
