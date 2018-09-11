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

import static java.util.Comparator.comparing;

import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.util.Context;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.net.URI;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarOutputStream;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileManager.Location;
import javax.tools.JavaFileObject;
import javax.tools.JavaFileObject.Kind;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;

/**
 * Output a jar file containing all classes on the platform classpath of the given JDK release.
 *
 * <p>usage: DumpPlatformClassPath <release version> <output jar>
 */
public class DumpPlatformClassPath {

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("usage: DumpPlatformClassPath <release version> <output jar>");
      System.exit(1);
    }
    int release = Integer.parseInt(args[0]);
    Path output = Paths.get(args[1]);

    Map<String, byte[]> entries = new HashMap<>();

    // Legacy JDK 8 bootclasspath handling.
    // TODO(cushon): make sure this has test coverage.
    Path javaHome = Paths.get(System.getProperty("java.home"));
    if (javaHome.endsWith("jre")) {
      javaHome = javaHome.getParent();
    }

    List<Path> jars = new ArrayList<>();

    Path extDir = javaHome.resolve("jre/lib/ext");
    if (Files.exists(extDir)) {
      for (Path extJar : Files.newDirectoryStream(extDir, "*.jar")) {
        jars.add(extJar);
      }
    }

    for (String jar : Arrays.asList(
        "rt.jar",
        "resources.jar",
        "jsse.jar",
        "jce.jar",
        "charsets.jar")) {
      Path path = javaHome.resolve("jre/lib").resolve(jar);
      if (Files.exists(path)) {
        jars.add(path);
      }
    }

    for (Path path : jars) {
      try (JarFile jf = new JarFile(path.toFile())) {
        jf.stream()
            .forEachOrdered(
                entry -> {
                  try {
                    entries.put(entry.getName(), toByteArray(jf.getInputStream(entry)));
                  } catch (IOException e) {
                    throw new UncheckedIOException(e);
                  }
                });
      }
    }

    if (!entries.isEmpty()) {
      // If we found a JDK 8 bootclasspath (rt.jar, etc.) then we're done.
      //
      // However JDK 8 only contains bootclasspath API information for the current release,
      // so we're always going to get a JDK 8 API level regardless of what the user requested.
      // Emit a warning if they wanted to target a different version.
      if (release != 8) {
        System.err.printf(
            "warning: ignoring release %s on --host_javabase=%s\n",
            release, System.getProperty("java.version"));
      }
    } else {
      // JDK > 8 --host_javabase bootclasspath handling.
      // The default --host_javabase is currently JDK 10.

      // Set up a compilation with --release to initialize a filemanager
      Context context = new Context();
      JavacTool.create()
          .getTask(
              /* out = */ null,
              /* fileManager = */ null,
              /* diagnosticListener = */ null,
              /* options = */ Arrays.asList("--release", String.valueOf(release)),
              /* classes = */ null,
              /* compilationUnits = */ null,
              context);
      StandardJavaFileManager fileManager =
          (StandardJavaFileManager) context.get(JavaFileManager.class);

      if (isJdk9OrEarlier()) {
        for (Path path : getLocationAsPaths(fileManager)) {
          Files.walkFileTree(
              path,
              new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
                    throws IOException {
                  if (file.getFileName().toString().endsWith(".sig")) {
                    String outputPath = path.relativize(file).toString();
                    outputPath =
                        outputPath.substring(0, outputPath.length() - ".sig".length()) + ".class";
                    entries.put(outputPath, Files.readAllBytes(file));
                  }
                  return FileVisitResult.CONTINUE;
                }
              });
        }
      } else {
        for (JavaFileObject fileObject :
            fileManager.list(
                StandardLocation.PLATFORM_CLASS_PATH,
                "",
                EnumSet.of(Kind.CLASS),
                /* recurse= */ true)) {
          String binaryName =
              fileManager.inferBinaryName(StandardLocation.PLATFORM_CLASS_PATH, fileObject);
          entries.put(
              binaryName.replace('.', '/') + ".class", toByteArray(fileObject.openInputStream()));
        }
      }

      // Include the jdk.unsupported module for compatibility with JDK 8.
      // (see: https://bugs.openjdk.java.net/browse/JDK-8206937)
      // `--release 8` only provides access to supported APIs, which excludes e.g. sun.misc.Unsafe.
      Path module =
          FileSystems.getFileSystem(URI.create("jrt:/")).getPath("modules/jdk.unsupported");
      Files.walkFileTree(
          module,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                throws IOException {
              String name = path.getFileName().toString();
              if (name.endsWith(".class") && !name.equals("module-info.class")) {
                entries.put(module.relativize(path).toString(), Files.readAllBytes(path));
              }
              return super.visitFile(path, attrs);
            }
          });
    }

    try (OutputStream os = Files.newOutputStream(output);
        BufferedOutputStream bos = new BufferedOutputStream(os, 65536);
        JarOutputStream jos = new JarOutputStream(bos)) {
      entries.entrySet().stream()
          .sorted(comparing(Map.Entry::getKey))
          .forEachOrdered(
              entry -> {
                try {
                  addEntry(jos, entry.getKey(), entry.getValue());
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
              });
    }
  }

  // Use a fixed timestamp for deterministic jar output.
  private static final long FIXED_TIMESTAMP =
      new GregorianCalendar(2010, 0, 1, 0, 0, 0).getTimeInMillis();

  private static void addEntry(JarOutputStream jos, String name, byte[] bytes) throws IOException {
    JarEntry je = new JarEntry(name);
    je.setTime(FIXED_TIMESTAMP);
    je.setMethod(ZipEntry.STORED);
    je.setSize(bytes.length);
    CRC32 crc = new CRC32();
    crc.update(bytes);
    je.setCrc(crc.getValue());
    jos.putNextEntry(je);
    jos.write(bytes);
  }

  private static byte[] toByteArray(InputStream is) throws IOException {
    byte[] buffer = new byte[8192];
    ByteArrayOutputStream boas = new ByteArrayOutputStream();
    while (true) {
      int r = is.read(buffer);
      if (r == -1) {
        break;
      }
      boas.write(buffer, 0, r);
    }
    return boas.toByteArray();
  }

  @SuppressWarnings("unchecked")
  private static Iterable<Path> getLocationAsPaths(StandardJavaFileManager fileManager) {
    try {
      return (Iterable<Path>)
          StandardJavaFileManager.class
              .getMethod("getLocationAsPaths", Location.class)
              .invoke(fileManager, StandardLocation.PLATFORM_CLASS_PATH);
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage(), e);
    }
  }

  static boolean isJdk9OrEarlier() {
    try {
      Method versionMethod = Runtime.class.getMethod("version");
      Object version = versionMethod.invoke(null);
      int majorVersion = (int) version.getClass().getMethod("major").invoke(version);
      return majorVersion <= 9;
    } catch (ReflectiveOperationException e) {
      return true;
    }
  }
}
