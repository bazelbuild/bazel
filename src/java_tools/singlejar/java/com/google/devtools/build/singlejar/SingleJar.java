// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.singlejar;

import static com.google.devtools.build.singlejar.ZipCombiner.DOS_EPOCH;

import com.google.devtools.build.singlejar.DefaultJarEntryFilter.PathFilter;
import com.google.devtools.build.singlejar.ZipCombiner.OutputMode;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import javax.annotation.concurrent.NotThreadSafe;

/**
 * An application that emulates the existing SingleJar tool, using the {@link
 * ZipCombiner} class.
 */
@NotThreadSafe
public class SingleJar {

  private static final byte NEWLINE_BYTE = (byte) '\n';
  private static final String MANIFEST_FILENAME = JarFile.MANIFEST_NAME;
  private static final String BUILD_DATA_FILENAME = "build-data.properties";

  private final SimpleFileSystem fileSystem;

  /** The input jar files we want to combine into the output jar. */
  private final List<String> inputJars = new ArrayList<>();

  /** Additional resources to be added to the output jar. */
  private final List<String> resources = new ArrayList<>();

  /** Additional class path resources to be added to the output jar. */
  private final List<String> classpathResources = new ArrayList<>();

  /** The name of the output Jar file. */
  private String outputJar;

  /** A filter for what jar entries to include */
  private PathFilter allowedPaths = DefaultJarEntryFilter.ANY_PATH;

  /** Extra manifest contents. */
  private String extraManifestContent;
  /** The main class - this is put into the manifest and also into the build info. */
  private String mainClass;

  /**
   * Warn about duplicate resource files, and skip them. Default behavior is to
   * give an error message.
   */
  private boolean warnDuplicateFiles = false;

  /** Indicates whether to set all timestamps to a fixed value. */
  private boolean normalize = false;
  private boolean checkDesugarDeps = false;
  private OutputMode outputMode = OutputMode.FORCE_STORED;

  /** Whether to include build-data.properties file */
  protected boolean includeBuildData = true;

  /** List of build information properties files */
  protected List<String> buildInformationFiles = new ArrayList<>();

  /** Extraneous build informations (key=value) */
  protected List<String> buildInformations = new ArrayList<>();

  /** The (optional) native executable that will be prepended to this JAR. */
  private String launcherBin = null;

  // Only visible for testing.
  protected SingleJar(SimpleFileSystem fileSystem) {
    this.fileSystem = fileSystem;
  }

  /**
   * Creates a manifest and returns an input stream for its contents.
   */
  private InputStream createManifest() throws IOException {
    Manifest manifest = new Manifest();
    Attributes attributes = manifest.getMainAttributes();
    attributes.put(Attributes.Name.MANIFEST_VERSION, "1.0");
    attributes.put(new Attributes.Name("Created-By"), "blaze-singlejar");
    if (mainClass != null) {
      attributes.put(Attributes.Name.MAIN_CLASS, mainClass);
    }
    if (extraManifestContent != null) {
      ByteArrayInputStream in = new ByteArrayInputStream(extraManifestContent.getBytes("UTF8"));
      manifest.read(in);
    }
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    manifest.write(out);
    return new ByteArrayInputStream(out.toByteArray());
  }

  private InputStream createBuildData() throws IOException {
    Properties properties = mergeBuildData();
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    properties.store(outputStream, null);
    byte[] output = outputStream.toByteArray();
    // Properties#store() adds a timestamp comment as first line, delete it.
    output = stripFirstLine(output);
    return new ByteArrayInputStream(output);
  }

  static byte[] stripFirstLine(byte[] output) {
    int i = 0;
    while (i < output.length && output[i] != NEWLINE_BYTE) {
      i++;
    }
    if (i < output.length) {
      output = Arrays.copyOfRange(output, i + 1, output.length);
    } else {
      output = new byte[0];
    }
    return output;
  }

  private Properties mergeBuildData() throws IOException {
    Properties properties = new Properties();
    for (String fileName : buildInformationFiles) {
      InputStream file = fileSystem.getInputStream(fileName);
      if (file != null) {
        properties.load(file);
      }
    }

    // extra properties
    for (String info : buildInformations) {
      String[] split = info.split("=", 2);
      String key = split[0];
      String value = "";
      if (split.length > 1) {
        value = split[1];
      }
      properties.put(key, value);
    }

    // finally add generic information
    // TODO(b/28294322): do we need to resolve the path to be absolute or canonical?
    properties.put("build.target", outputJar);
    if (mainClass != null) {
      properties.put("main.class", mainClass);
    }
    return properties;
  }

  private String getName(String filename) {
    int index = filename.lastIndexOf('/');
    return index < 0 ? filename : filename.substring(index + 1);
  }

  // Only visible for testing.
  protected int run(List<String> args) throws IOException {
    List<String> expandedArgs = new OptionFileExpander(fileSystem).expandArguments(args);
    processCommandlineArgs(expandedArgs);
    InputStream buildInfo = createBuildData();

    ZipCombiner combiner = null;
    try (OutputStream out = fileSystem.getOutputStream(outputJar)) {
      combiner = new ZipCombiner(outputMode, createEntryFilterHelper(), out);
      if (launcherBin != null) {
        combiner.prependExecutable(fileSystem.getInputStream(launcherBin));
      }
      Date date = normalize ? ZipCombiner.DOS_EPOCH : null;

      // Add a manifest file.
      JarUtils.addMetaInf(combiner, date);
      combiner.addFile(MANIFEST_FILENAME, date, createManifest());

      if (includeBuildData) {
        // Add the build data file.
        combiner.addFile(BUILD_DATA_FILENAME, date, buildInfo);
      }

      // Copy the resources to the top level of the jar file.
      for (String classpathResource : classpathResources) {
        String entryName = getName(classpathResource);
        if (warnDuplicateFiles && combiner.containsFile(entryName)) {
          System.err.println("File " + entryName + " clashes with a previous file");
          continue;
        }
        combiner.addFile(entryName, date, fileSystem.getInputStream(classpathResource));
      }

      // Copy the resources into the jar file.
      for (String resource : resources) {
        String from;
        String to;
        int i = resource.indexOf(':');
        if (i < 0) {
          to = from = resource;
        } else {
          from = resource.substring(0, i);
          to = resource.substring(i + 1);
        }
        if (warnDuplicateFiles && combiner.containsFile(to)) {
          System.err.println("File " + from + " at " + to + " clashes with a previous file");
          continue;
        }

        // Add parent directory entries.
        int idx = to.indexOf('/');
        while (idx != -1) {
          String dir = to.substring(0, idx + 1);
          if (!combiner.containsFile(dir)) {
            combiner.addDirectory(dir, DOS_EPOCH);
          }
          idx = to.indexOf('/', idx + 1);
        }

        combiner.addFile(to, date, fileSystem.getInputStream(from));
      }

      // Copy the jars into the jar file.
      for (String inputJar : inputJars) {
        File jar = fileSystem.getFile(inputJar);
        combiner.addZip(jar);
      }

      // Close the output file. If something goes wrong here, delete the file.
      combiner.close();
      combiner = null;
    } finally {
      // This part is only executed if an exception occurred.
      if (combiner != null) {
        try {
          // We may end up calling close twice, but that's ok.
          combiner.close();
        } catch (IOException e) {
          // There's already an exception in progress - this won't add any
          // additional information.
        }
        // Ignore return value - there's already an exception in progress.
        fileSystem.delete(outputJar);
      }
    }
    return 0;
  }

  private ZipEntryFilter createEntryFilterHelper() {
    ZipEntryFilter result = createEntryFilter(normalize, allowedPaths);
    if (checkDesugarDeps) {
      // Invocation is done through reflection so that this code will work in bazel open source
      // as well. SingleJar is used for bootstrap and thus can not depend on protos (used in
      // Java8DesugarDepsJarEntryFilter).
      try {
        return (ZipEntryFilter)
            Class.forName("com.google.devtools.build.singlejar.Java8DesugarDepsJarEntryFilter")
                .getConstructor(ZipEntryFilter.class).newInstance(result);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException("Couldn't instantiate desugar deps checker", e);
      }
    } else {
      return (filename, callback) -> {
        if ("META-INF/desugar_deps".equals(filename)) {
          callback.skip();  // We never want these files in the output
        } else {
          result.accept(filename, callback);
        }
      };
    }

  }

  protected ZipEntryFilter createEntryFilter(boolean normalize, PathFilter allowedPaths) {
    return new DefaultJarEntryFilter(normalize, allowedPaths);
  }

  /**
   * Collects the arguments for a command line flag until it finds a flag that
   * starts with the terminatorPrefix.
   *
   * @param args
   * @param startIndex the start index in the args to collect the flag arguments
   *        from
   * @param flagArguments the collected flag arguments
   * @param terminatorPrefix the terminator prefix to stop collecting of
   *        argument flags
   * @return the index of the first argument that started with the
   *         terminatorPrefix
   */
  private static int collectFlagArguments(List<String> args, int startIndex,
      List<String> flagArguments, String terminatorPrefix) {
    startIndex++;
    while (startIndex < args.size()) {
      String name = args.get(startIndex);
      if (name.startsWith(terminatorPrefix)) {
        return startIndex - 1;
      }
      flagArguments.add(name);
      startIndex++;
    }
    return startIndex;
  }

  /**
   * Returns a single argument for a command line option.
   *
   * @throws IOException if no more arguments are available
   */
  private static String getArgument(List<String> args, int i, String arg) throws IOException {
    if (i + 1 < args.size()) {
      return args.get(i + 1);
    }
    throw new IOException(arg + ": missing argument");
  }

  /**
   * Processes the command line arguments.
   *
   * @throws IOException if one of the files containing options cannot be read
   */
  protected void processCommandlineArgs(List<String> args) throws IOException {
    List<String> manifestLines = new ArrayList<>();
    List<String> prefixes = new ArrayList<>();
    for (int i = 0; i < args.size(); i++) {
      String arg = args.get(i);
      if (arg.equals("--sources")) {
        i = collectFlagArguments(args, i, inputJars, "--");
      } else if (arg.equals("--resources")) {
        i = collectFlagArguments(args, i, resources, "--");
      } else if (arg.equals("--classpath_resources")) {
        i = collectFlagArguments(args, i, classpathResources, "--");
      } else if (arg.equals("--deploy_manifest_lines")) {
        i = collectFlagArguments(args, i, manifestLines, "--");
      } else if (arg.equals("--build_info_file")) {
        buildInformationFiles.add(getArgument(args, i, arg));
        i++;
      } else if (arg.equals("--extra_build_info")) {
        buildInformations.add(getArgument(args, i, arg));
        i++;
      } else if (arg.equals("--main_class")) {
        mainClass = getArgument(args, i, arg);
        i++;
      } else if (arg.equals("--output")) {
        outputJar = getArgument(args, i, arg);
        i++;
      } else if (arg.equals("--compression")) {
        outputMode = OutputMode.FORCE_DEFLATE;
      } else if (arg.equals("--dont_change_compression")) {
        outputMode = OutputMode.DONT_CARE;
      } else if (arg.equals("--normalize")) {
        normalize = true;
      } else if (arg.equals("--include_prefixes")) {
        i = collectFlagArguments(args, i, prefixes, "--");
      } else if (arg.equals("--exclude_build_data")) {
        includeBuildData = false;
      } else if (arg.equals("--warn_duplicate_resources")) {
        warnDuplicateFiles = true;
      } else if (arg.equals("--java_launcher")) {
        launcherBin = getArgument(args, i, arg);
        i++;
      } else if (arg.equals("--check_desugar_deps")) {
        checkDesugarDeps = true;
      } else {
        throw new IOException("unknown option : '" + arg + "'");
      }
    }
    if (!manifestLines.isEmpty()) {
      setExtraManifestContent(joinWithNewlines(manifestLines));
    }
    if (!prefixes.isEmpty()) {
      setPathPrefixes(prefixes);
    }
  }

  private String joinWithNewlines(Iterable<String> lines) {
    StringBuilder result = new StringBuilder();
    Iterator<String> it = lines.iterator();
    if (it.hasNext()) {
      result.append(it.next());
    }
    while (it.hasNext()) {
      result.append('\n');
      result.append(it.next());
    }
    return result.toString();
  }

  private void setExtraManifestContent(String extraManifestContent) {
    // The manifest content has to be terminated with a newline character
    if (!extraManifestContent.endsWith("\n")) {
      extraManifestContent = extraManifestContent + '\n';
    }
    this.extraManifestContent = extraManifestContent;
  }

  private void setPathPrefixes(List<String> prefixes) throws IOException {
    if (prefixes.isEmpty()) {
      throw new IOException(
          "Empty set of path prefixes; cowardly refusing to emit an empty jar file");
    }
    allowedPaths = new PrefixListPathFilter(prefixes);
  }

  static int singleRun(String[] args) throws IOException {
    SingleJar singlejar = new SingleJar(new JavaIoFileSystem());
    return singlejar.run(Arrays.asList(args));
  }

  public static void main(String[] args) {
    if (shouldRunInWorker(args)) {
      if (!canRunInWorker()) {
        System.err.println("Asked to run in a worker, but no worker support");
        System.exit(1);
      }
      try {
        runWorker(args);
      } catch (Exception e) {
        System.err.println("Error running worker : " + e.getMessage());
        System.exit(1);
      }
      return;
    }

    try {
      System.exit(singleRun(args));
    } catch (IOException e) {
      System.err.println("SingleJar threw exception : " + e.getMessage());
      System.exit(1);
    }
  }

  private static void runWorker(String[] args) throws Exception {
    // Invocation is done through reflection so that this code will work in bazel open source
    // as well. SingleJar is used for bootstrap and thus can not depend on protos (used in
    // SingleJarWorker).
    Class<?> workerClass = Class.forName("com.google.devtools.build.singlejar.SingleJarWorker");
    workerClass.getMethod("main", String[].class).invoke(null, (Object) args);
  }

  protected static boolean shouldRunInWorker(String[] args) {
    return Arrays.asList(args).contains("--persistent_worker");
  }

  private static boolean canRunInWorker() {
    try {
      Class.forName("com.google.devtools.build.singlejar.SingleJarWorker");
      return true;
    } catch (ClassNotFoundException e1) {
      return false;
    }
  }
  
}
