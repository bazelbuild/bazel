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
package com.google.devtools.build.android.ziputils;

import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTCRC;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTLEN;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENCRC;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENTIM;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCFLG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCTIM;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.common.collect.Sets;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

/**
 * Extracts entries from a set of input archives, and copies them to N output archive of
 * approximately equal size, while attempting to split archives on package (directory) boundaries.
 * Optionally, accept a list of entries to be added to the first output archive, splitting
 * remaining entries by package boundaries.
 */
public class SplitZip implements EntryHandler {
  private boolean verbose = false;
  private boolean splitDexFiles = false;
  private final List<ZipIn> inputs;
  private final List<ZipOut> outputs;
  private String filterFile;
  private InputStream filterInputStream;
  private String resourceFile;
  private Date date;
  private DosTime dosTime;
  // Internal state variables:
  private boolean finished = false;
  private Set<String> filter;
  private ZipOut[] zipOuts;
  private ZipOut resourceOut;
  private final Map<String, ZipOut> assignments = new LinkedHashMap<>();
  private final Map<String, CentralDirectory> centralDirectories;
  private final Set<String> classes = new TreeSet<>();
  private Predicate<String> inputFilter = Predicates.alwaysTrue();

  /**
   * Creates an un-configured {@code SplitZip} instance.
   */
  public SplitZip() {
    inputs = new ArrayList<>();
    outputs = new ArrayList<>();
    centralDirectories = new LinkedHashMap<>();
  }

  /**
   * Configures a resource file. By default, resources are output in the initial shard. If a
   * resource file is specified, resources are written to this instead.
   *
   * @param resourceFile in not {@code null}, the name of a file in which to output resources.
   * @return this object.
   */
  @CanIgnoreReturnValue
  public SplitZip setResourceFile(String resourceFile) {
    this.resourceFile = resourceFile;
    return this;
  }

  // Package private for testing with mock file
  @CanIgnoreReturnValue
  SplitZip setResourceFile(ZipOut resOut) {
    resourceOut = resOut;
    return this;
  }

  /**
   * Gets the name of the resource output file. If no resource output file is configured, resources
   * are output in the initial shard.
   * @return the name of the resource output file, or {@code null} if no file has been configured.
   */
  public String getResourceFile() {
    return resourceFile;
  }

  /**
   * Configures a file containing a list of files to be included in the first output archive.
   *
   * @param clFile path of class file list.
   * @return this object
   */
  @CanIgnoreReturnValue
  public SplitZip setMainClassListFile(String clFile) {
    filterFile = clFile;
    return this;
  }

  // Package private for testing with mock file
  @CanIgnoreReturnValue
  SplitZip setMainClassListStreamForTesting(InputStream clInputStream) {
    filterInputStream = clInputStream;
    return this;
  }

  /**
   * Gets the path of the file listing the content of the initial shard.
   * @return return path of file list file, or {@code null} if not set.
   */
  public String getMainClassListFile() {
    return filterFile;
  }

  /**
   * Configures verbose mode.
   *
   * @param flag set to {@code true} to turn on verbose mode.
   * @return this object
   */
  @CanIgnoreReturnValue
  public SplitZip setVerbose(boolean flag) {
    verbose = flag;
    return this;
  }

  /**
   * Gets the verbosity mode.
   * @return {@code true} iff verbose mode is enabled
   */
  public boolean isVerbose() {
    return verbose;
  }

  /**
   * Configures whether to split .dex files along with .class files.
   *
   * @param flag {@code true} will split .dex files; {@code false} treats them as resources
   */
  @CanIgnoreReturnValue
  public SplitZip setSplitDexedClasses(boolean flag) {
    splitDexFiles = flag;
    return this;
  }

  /**
   * Sets date to overwrite timestamp of copied entries. Setting the date to {@code null} means
   * using the date and time information in the input file. Set an explicit date to override.
   *
   * @param date modified date and time to set for entries in output.
   * @return this object.
   */
  @CanIgnoreReturnValue
  public SplitZip setEntryDate(Date date) {
    this.date = date;
    this.dosTime = date == null ? null : new DosTime(date);
    return this;
  }

  /**
   * Sets date to {@link DosTime#DOS_EPOCHISH}.
   *
   * @return this object.
   */
  @CanIgnoreReturnValue
  public SplitZip useDefaultEntryDate() {
    this.date = DosTime.DOS_EPOCHISH;
    this.dosTime = DosTime.EPOCHISH;
    return this;
  }

  /**
   * Gets the entry modified date.
   */
  public Date getEntryDate() {
    return date;
  }

  /**
   * Configures multiple input file locations.
   *
   * @param inputs list of input locations.
   * @return this object
   * @throws java.io.IOException
   */
  @CanIgnoreReturnValue
  public SplitZip addInputs(Iterable<String> inputs) throws IOException {
    for (String i : inputs) {
      addInput(i);
    }
    return this;
  }

  /**
   * Configures an input location. An input file must be a zip archive.
   *
   * @param filename path for an input location.
   * @return this object
   * @throws java.io.IOException
   */
  @CanIgnoreReturnValue
  public SplitZip addInput(String filename) throws IOException {
    if (filename != null) {
      inputs.add(new ZipIn(new FileInputStream(filename).getChannel(), filename));
    }
    return this;
  }

  // Package private, for testing using mock file system.
  @CanIgnoreReturnValue
  SplitZip addInput(ZipIn in) throws IOException {
    Preconditions.checkNotNull(in);
    inputs.add(in);
    return this;
  }

  /**
   * Configures multiple output file locations.
   *
   * @param outputs list of output files.
   * @return this object
   * @throws java.io.IOException
   */
  @CanIgnoreReturnValue
  public SplitZip addOutputs(Iterable<String> outputs) throws IOException {
    for (String o : outputs) {
      addOutput(o);
    }
    return this;
  }

  /**
   * Configures an output location.
   *
   * @param output path for an output location.
   * @return this object
   * @throws java.io.IOException
   */
  @CanIgnoreReturnValue
  public SplitZip addOutput(String output) throws IOException {
    Preconditions.checkNotNull(output);
    outputs.add(new ZipOut(new FileOutputStream(output, false).getChannel(), output));
    return this;
  }

  // Package private for testing with mock file
  @CanIgnoreReturnValue
  SplitZip addOutput(ZipOut output) throws IOException {
    Preconditions.checkNotNull(output);
    outputs.add(output);
    return this;
  }

  /**
   * Set a predicate to only include files with matching filenames in any of the outputs. <b>Other
   * zip entries are dropped</b>, regardless of whether they're classes or resources and regardless
   * of whether they're listed in {@link #setMainClassListFile}.
   */
  @CanIgnoreReturnValue
  public SplitZip setInputFilter(Predicate<String> inputFilter) {
    this.inputFilter = Preconditions.checkNotNull(inputFilter);
    return this;
  }

  /**
   * Executes this {@code SplitZip}, reading content from the configured input locations, creating
   * the specified number of archives, in the configured output directory.
   *
   * @return this object
   * @throws java.io.IOException
   */
  @CanIgnoreReturnValue
  public SplitZip run() throws IOException {
    verbose("SplitZip: Splitting in: " + outputs.size());
    verbose("SplitZip: with filter: " + filterFile);
    checkConfig();
    // Prepare output files
    zipOuts = outputs.toArray(new ZipOut[outputs.size()]);
    if (resourceFile != null) {
      resourceOut = new ZipOut(new FileOutputStream(resourceFile, false).getChannel(),
          resourceFile);
    } else if (resourceOut == null) { // may have been set for testing
      resourceOut = zipOuts[0];
    }

    // Read directories of input files
    for (ZipIn zip : inputs) {
      zip.endOfCentralDirectory();
      centralDirectories.put(zip.getFilename(), zip.centralDirectory());
      zip.centralDirectory();
    }
    // Assign input entries to output files
    split();
    // Copy entries to the assigned output files
    for (ZipIn zip : inputs) {
      zip.scanEntries(this);
    }
    return this;
  }

  /**
   * Copies an entry to the assigned output files. Called for each entry in the input files.
   * @param in
   * @param header
   * @param dirEntry
   * @param data
   * @throws IOException
   */
  @Override
  public void handle(ZipIn in, LocalFileHeader header, DirectoryEntry dirEntry,
      ByteBuffer data) throws IOException {
    ZipOut out = assignments.remove(normalizedFilename(header.getFilename()));
    if (out == null) {
      // Skip unassigned file; includes a file with the same name as a previously processed one.
      // This in particular picks the first .class or .dex file encountered for a given class name
      // and drops any file not matched by inputFilter.
      return;
    }
    if (dirEntry == null) {
      // Shouldn't get here, as there should be no assignment.
      System.out.println("Warning: no directory entry");
      return;
    }
    // Clone directory entry
    DirectoryEntry entryOut = out.nextEntry(dirEntry);
    if (dosTime != null) {
      // Overwrite time stamp
      header.set(LOCTIM, dosTime.time);
      entryOut.set(CENTIM, dosTime.time);
    }
    out.write(header);
    out.write(data);
    if ((header.get(LOCFLG) & LocalFileHeader.SIZE_MASKED_FLAG) != 0) {
      // Instead of this, we could fix the header with the size information
      // from the directory entry. For now, keep the entry encoded as-is.
      DataDescriptor desc = DataDescriptor.allocate()
          .set(EXTCRC, dirEntry.get(CENCRC))
          .set(EXTSIZ, dirEntry.get(CENSIZ))
          .set(EXTLEN, dirEntry.get(CENLEN));
      out.write(desc);
    }
  }

  /**
   * Writes any remaining output data to the output stream.
   *
   * @throws IOException if the output stream or the filter throws an IOException
   * @throws IllegalStateException if this method was already called earlier
   */
  public void finish() throws IOException {
    checkNotFinished();
    finished = true;
    if (resourceOut != null) {
      resourceOut.finish();
    }
    for (ZipOut zo : zipOuts) {
      zo.finish();
    }
  }

  /**
   * Writes any remaining output data to the output stream and closes it.
   *
   * @throws IOException if the output stream or the filter throws an IOException
   */
  public void close() throws IOException {
    if (!finished) {
      finish();
    }
    if (resourceOut != null) {
      resourceOut.close();
    }
    for (ZipOut zo : zipOuts) {
      zo.close();
    }
  }

  private void checkNotFinished() {
    if (finished) {
      throw new IllegalStateException();
    }
  }

  /**
   * Validates configuration before execution.
   */
  private void checkConfig() throws IOException {
    if (outputs.size() < 1) {
      throw new IllegalStateException("Require at least one output file");
    }
    filter = filterFile == null && filterInputStream == null ? null : readPaths(filterFile);
  }

  /** Parses the entries and assign each entry to an output file. */
  private void split() throws IOException {

    // A map of class (a "context") to its inner synthetic classes from D8.
    SetMultimap<String, String> syntheticClassContexts = HashMultimap.create();

    for (ZipIn in : inputs) {
      CentralDirectory cdir = centralDirectories.get(in.getFilename());

      for (DirectoryEntry entry : cdir.list()) {
        if (entry.getFilename().equals("META-INF/synthetic-contexts.map")) {
          parseSyntheticContextsMap(in.entryFor(entry).getContent(), syntheticClassContexts);
          break;
        }
      }

      for (DirectoryEntry entry : cdir.list()) {
        String filename = normalizedFilename(entry.getFilename());
        if (!inputFilter.apply(filename)) {
          continue;
        }
        if (filename.endsWith(".class")) {
          // Only pass classes to the splitter, so that it can do the best job
          // possible distributing them across output files.
          classes.add(filename);
        } else if (!filename.endsWith("/")) {
          // Non class files (resources) are either assigned to the first
          // output file, or to a specified resource output file.
          assignments.put(filename, resourceOut);
        }
      }
    }

    Splitter splitter = new Splitter(outputs.size(), classes.size());

    if (filter != null) {
      // Assign files in the filter to the first output file.
      splitter.assignAllToCurrentShard(Sets.filter(filter, inputFilter));
      splitter.nextShard(); // minimal initial shard
    }

    Set<String> allSyntheticClasses = new HashSet<>(syntheticClassContexts.values());

    for (String path : classes) {
      if (!allSyntheticClasses.contains(path)) {

        // Use normalized filename so the filter file doesn't have to change
        int assignment = splitter.assign(path);
        Set<String> syntheticClasses = syntheticClassContexts.get(path);
        splitter.assignAllToCurrentShard(syntheticClasses);
        Preconditions.checkState(assignment >= 0 && assignment < zipOuts.length);

        assignments.put(path, zipOuts[assignment]);
        for (String syntheticClass : syntheticClasses) {
          assignments.put(syntheticClass, zipOuts[assignment]);
        }
      }
    }
  }

  private String normalizedFilename(String filename) {
    if (splitDexFiles && filename.endsWith(".class.dex")) { // suffix generated by DexBuilder
      return filename.substring(0, filename.length() - ".dex".length());
    }
    return filename;
  }

  /**
   * Reads paths of classes required in first shard. For testing purposes, this relies
   * on the file system configured for the {@code Zip} library class.
   */
  private Set<String> readPaths(String fileName) throws IOException {
    Set<String> paths = new LinkedHashSet<>();
    if (filterInputStream == null) {
      filterInputStream = new FileInputStream(fileName);
    }
    try (BufferedReader reader =
        new BufferedReader(new InputStreamReader(filterInputStream, UTF_8))) {
      String line;
      while (null != (line = reader.readLine())) {
        paths.add(fixPath(line));
      }
      return paths;
    }
  }

  // TODO(bazel-team): Got this from 'dx'. I'm not sure we need this part. Keep it for now,
  // to make sure we read the main dex list the exact same way that dx would.
  private String fixPath(String path) {
    if (File.separatorChar == '\\') {
      path = path.replace('\\', '/');
    }
    int index = path.lastIndexOf("/./");
    if (index != -1) {
      return path.substring(index + 3);
    }
    if (path.startsWith("./")) {
      return path.substring(2);
    }
    return path;
  }

  private static void parseSyntheticContextsMap(
      ByteBuffer byteBuffer, Multimap<String, String> syntheticClassContexts) {
    // The ByteBuffer returned from the Splitter's zip library is not backed by an accessible array,
    // so ByteBuffer.array() is not supported, so we must go the long way.
    byte[] bytes = new byte[byteBuffer.remaining()];
    byteBuffer.get(bytes);
    Scanner scanner = new Scanner(new ByteArrayInputStream(bytes), UTF_8);
    scanner.useDelimiter("[;\n]");
    while (scanner.hasNext()) {
      String syntheticClass = scanner.next();
      String context = scanner.next();
      // The context map uses class names, whereas SplitZip uses class file names, so add ".class"
      // here to make it easier to work with the map in the rest of the code.
      syntheticClassContexts.put(context + ".class", syntheticClass + ".class");
    }
  }

  private void verbose(String msg) {
    if (verbose) {
      System.out.println(msg);
    }
  }
}
