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

import com.sun.tools.javac.api.JavacTool;
import com.sun.tools.javac.util.Context;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
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
 * <p>usage: DumpPlatformClassPath <release version> <output jar> <path to target JDK>?
 */
public class DumpPlatformClassPath {

  public static void main(String[] args) throws Exception {
    if (args.length < 2 || args.length > 3) {
      System.err.println(
          "usage: DumpPlatformClassPath <release version> <output jar> <path to target JDK>?");
      System.exit(1);
    }
    int release = Integer.parseInt(args[0]);
    Path output = Paths.get(args[1]);
    Path targetJavabase = null;
    if (args.length == 3) {
      targetJavabase = Paths.get(args[2]);
    }

    int hostMajorVersion = hostMajorVersion();
    boolean ok;
    if (hostMajorVersion == 8) {
      ok = dumpJDK8BootClassPath(release, output, targetJavabase);
    } else {
      ok = dumpJDK9AndNewerBootClassPath(hostMajorVersion, release, output, targetJavabase);
    }
    System.exit(ok ? 0 : 1);
  }

  // JDK 8 bootclasspath handling.
  // * JDK 8 represents a bootclasspath as a search path of jars (rt.jar, etc.).
  // * It does not support --release or --system.
  static boolean dumpJDK8BootClassPath(int release, Path output, Path targetJavabase)
      throws IOException {
    if (release != 8) {
      System.err.printf("error: --release=%s is not supported on --host_javabase=8\n", release);
      return false;
    }
    List<Path> bootClassPathJars;
    if (targetJavabase != null) {
      bootClassPathJars = getBootClassPathJars(targetJavabase);
    } else {
      Path hostJavabase = Paths.get(System.getProperty("java.home"));
      if (hostJavabase.endsWith("jre")) {
        hostJavabase = hostJavabase.getParent();
      }
      bootClassPathJars = getBootClassPathJars(hostJavabase);
    }
    writeClassPathJars(output, bootClassPathJars);
    return true;
  }

  // JDK > 8 --host_javabase bootclasspath handling.
  // (The default --host_javabase is currently JDK 9.)
  static boolean dumpJDK9AndNewerBootClassPath(
      int hostMajorVersion, int release, Path output, Path targetJavabase) throws IOException {

    // JDK 9 and newer support cross-compiling to older platform versions using the --system
    // and --release flags.
    // * --system takes the path to a JDK root for JDK 9 and up, and causes the compilation
    //     to target the APIs from that JDK.
    // * --release takes a language level (e.g. '9') and uses the API information baked in to
    //     the host JDK (in lib/ct.sym).

    // Since --system only supports JDK >= 9, first check of the target JDK defines a JDK 8
    // bootclasspath.
    if (targetJavabase != null) {
      List<Path> bootClassPathJars = getBootClassPathJars(targetJavabase);
      if (!bootClassPathJars.isEmpty()) {
        writeClassPathJars(output, bootClassPathJars);
        return true;
      }
      if (release == 8) {
        System.err.printf(
            "warning: could not find a JDK 8 bootclasspath in %s, falling back to --release\n",
            targetJavabase);
      }
    }

    // Initialize a FileManager to process the --release or --system arguments, and then read the
    // initialized bootclasspath data back out.

    List<String> javacOptions =
        targetJavabase != null
            ? Arrays.asList("--system", String.valueOf(targetJavabase))
            : Arrays.asList("--release", String.valueOf(release));

    Context context = new Context();
    JavacTool.create()
        .getTask(
            /* out = */ null,
            /* fileManager = */ null,
            /* diagnosticListener = */ null,
            /* options = */ javacOptions,
            /* classes = */ null,
            /* compilationUnits = */ null,
            context);
    StandardJavaFileManager fileManager =
        (StandardJavaFileManager) context.get(JavaFileManager.class);

    SortedMap<String, InputStream> entries = new TreeMap<>();
    if (hostMajorVersion == 9 && targetJavabase == null && release == 8) {
      // Work-around: when running on a JDK 9 host_javabase with --release 8, the ct.sym
      // handling isn't compatible with the FileManager#list code path in the branch below.
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
                  entries.put(outputPath, Files.newInputStream(file));
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
        entries.put(binaryName.replace('.', '/') + ".class", fileObject.openInputStream());
      }
    }
    writeEntries(output, entries);
    return true;
  }

  /** Writes the given entry names and data to a jar archive at the given path. */
  private static void writeEntries(Path output, Map<String, InputStream> entries)
      throws IOException {
    if (!entries.containsKey("java/lang/Object.class")) {
      throw new AssertionError(
          "\nCould not find java.lang.Object on bootclasspath; something has gone terribly wrong.\n"
              + "Please file a bug: https://github.com/bazelbuild/bazel/issues");
    }
    try (OutputStream os = Files.newOutputStream(output);
        BufferedOutputStream bos = new BufferedOutputStream(os, 65536);
        JarOutputStream jos = new JarOutputStream(bos)) {
      entries.entrySet().stream()
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

  /** Collects the entries of the given jar files into a map from jar entry names to their data. */
  private static void writeClassPathJars(Path output, Collection<Path> paths) throws IOException {
    List<JarFile> jars = new ArrayList<>();
    for (Path path : paths) {
      jars.add(new JarFile(path.toFile()));
    }
    SortedMap<String, InputStream> entries = new TreeMap<>();
    for (JarFile jar : jars) {
      jar.stream()
          .filter(p -> p.getName().endsWith(".class"))
          .forEachOrdered(
              entry -> {
                try {
                  entries.put(entry.getName(), jar.getInputStream(entry));
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
              });
    }
    writeEntries(output, entries);
    for (JarFile jar : jars) {
      jar.close();
    }
  }

  /** Returns paths to the entries of a JDK 8-style bootclasspath. */
  private static List<Path> getBootClassPathJars(Path javaHome) throws IOException {
    List<Path> jars = new ArrayList<>();
    Path extDir = javaHome.resolve("jre/lib/ext");
    if (Files.exists(extDir)) {
      for (Path extJar : Files.newDirectoryStream(extDir, "*.jar")) {
        jars.add(extJar);
      }
    }
    for (String jar :
        Arrays.asList("rt.jar", "resources.jar", "jsse.jar", "jce.jar", "charsets.jar")) {
      Path path = javaHome.resolve("jre/lib").resolve(jar);
      if (Files.exists(path)) {
        jars.add(path);
      }
    }
    return jars;
  }

  // Use a fixed timestamp for deterministic jar output.
  private static final long FIXED_TIMESTAMP =
      new GregorianCalendar(2010, 0, 1, 0, 0, 0).getTimeInMillis();

  /**
   * Add a jar entry to the given {@link JarOutputStream}, normalizing the entry timestamps to
   * ensure deterministic build output.
   */
  private static void addEntry(JarOutputStream jos, String name, InputStream input)
      throws IOException {
    JarEntry je = new JarEntry(name);
    je.setTime(FIXED_TIMESTAMP);
    je.setMethod(ZipEntry.STORED);
    byte[] bytes = toByteArray(input);
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

  /**
   * Reflectively calls {@code StandardJavaFileManager#getLocationAsPaths}, which is only available
   * in JDK 9 and newer.
   */
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

  /**
   * Returns the major version of the host Java runtime (e.g. '8' for JDK 8), using {@link
   * Runtime#version} if it is available, and otherwise falling back to the {@code
   * java.class.version} system. property.
   */
  static int hostMajorVersion() {
    try {
      Method versionMethod = Runtime.class.getMethod("version");
      Object version = versionMethod.invoke(null);
      return (int) version.getClass().getMethod("major").invoke(version);
    } catch (ReflectiveOperationException e) {
      // Runtime.version() isn't available on JDK 8; continue below
    }
    int version = (int) Double.parseDouble(System.getProperty("java.class.version"));
    if (49 <= version && version <= 52) {
      return version - (49 - 5);
    }
    throw new IllegalStateException(
        "Unknown Java version: " + System.getProperty("java.specification.version"));
  }
}
