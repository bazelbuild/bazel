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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;
import com.google.devtools.build.singlejar.ZipEntryFilter.StrategyCallback;
import com.google.devtools.build.zip.ExtraData;
import com.google.devtools.build.zip.ExtraDataList;
import com.google.devtools.build.zip.ZipFileEntry;
import com.google.devtools.build.zip.ZipFileEntry.Compression;
import com.google.devtools.build.zip.ZipReader;
import com.google.devtools.build.zip.ZipUtil;
import com.google.devtools.build.zip.ZipWriter;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.DeflaterInputStream;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;
import javax.annotation.Nullable;

/**
 * An object that combines multiple ZIP files into a single file. It only
 * supports a subset of the ZIP format, specifically:
 * <ul>
 *   <li>It only supports STORE and DEFLATE storage methods.</li>
 *   <li>It only supports 32-bit ZIP files.</li>
 * </ul>
 *
 * <p>These restrictions are also present in the JDK implementations
 * {@link java.util.jar.JarInputStream}, {@link java.util.zip.ZipInputStream},
 * though they are not documented there.
 *
 * <p>IMPORTANT NOTE: Callers must call {@link #finish()} or {@link #close()}
 * at the end of processing to ensure that the output buffers are flushed and
 * the ZIP file is complete.
 *
 * <p>This class performs only rudimentary data checking. If the input files
 * are damaged, the output will likely also be damaged.
 *
 * <p>Also see:
 * <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP format</a>
 */
public class ZipCombiner implements AutoCloseable {
  private static final int INFLATER_BUFFER_BYTES = 8192;
  public static final Date DOS_EPOCH = new Date(ZipUtil.DOS_EPOCH);
  /**
   * Whether to compress or decompress entries.
   */
  public enum OutputMode {

    /**
     * Output entries using any method.
     */
    DONT_CARE,

    /**
     * Output all entries using DEFLATE method, except directory entries. It is always more
     * efficient to store directory entries uncompressed.
     */
    FORCE_DEFLATE,

    /**
     * Output all entries using STORED method.
     */
    FORCE_STORED,
  }

  /**
   * The type of action to take for a ZIP file entry.
   */
  private enum ActionType {

    /**
     * Skip the entry.
     */
    SKIP,

    /**
     * Copy the entry.
     */
    COPY,

    /**
     * Rename the entry.
     */
    RENAME,

    /**
     * Merge the entry.
     */
    MERGE;
  }

  /**
   * Encapsulates the action to take for a ZIP file entry along with optional details specific to
   * the action type. The minimum requirements per type are:
   * <ul>
   *    <li>SKIP: none.</li>
   *    <li>COPY: none.</li>
   *    <li>RENAME: newName.</li>
   *    <li>MERGE: strategy, mergeBuffer.</li>
   * </ul>
   *
   * <p>An action can be easily changed from one type to another by using
   * {@link EntryAction#EntryAction(ActionType, EntryAction)}.
   */
  private static final class EntryAction {
    private final ActionType type;
    @Nullable private final Date date;
    @Nullable private final String newName;
    @Nullable private final CustomMergeStrategy strategy;
    @Nullable private final ByteArrayOutputStream mergeBuffer;

    /**
     * Create an action of the specified type with no extra details.
     */
    public EntryAction(ActionType type) {
      this(type, null, null, null, null);
    }

    /**
     * Create a duplicate action with a different {@link ActionType}.
     */
    public EntryAction(ActionType type, EntryAction action) {
      this(type, action.getDate(), action.getNewName(), action.getStrategy(),
          action.getMergeBuffer());
    }

    /**
     * Create an action of the specified type and details.
     *
     * @param type the type of action
     * @param date the custom date to set on the entry
     * @param newName the custom name to create the entry as
     * @param strategy the {@link CustomMergeStrategy} to use for merging this entry
     * @param mergeBuffer the output stream to use for merge results
     */
    public EntryAction(ActionType type, Date date, String newName, CustomMergeStrategy strategy,
        ByteArrayOutputStream mergeBuffer) {
      checkArgument(type != ActionType.RENAME || newName != null,
          "NewName must not be null if the ActionType is RENAME.");
      checkArgument(type != ActionType.MERGE || strategy != null,
          "Strategy must not be null if the ActionType is MERGE.");
      checkArgument(type != ActionType.MERGE || mergeBuffer != null,
          "MergeBuffer must not be null if the ActionType is MERGE.");
      this.type = type;
      this.date = date;
      this.newName = newName;
      this.strategy = strategy;
      this.mergeBuffer = mergeBuffer;
    }

    /** Returns the type. */
    public ActionType getType() {
      return type;
    }

    /** Returns the date. */
    public Date getDate() {
      return date;
    }

    /** Returns the new name. */
    public String getNewName() {
      return newName;
    }

    /** Returns the strategy. */
    public CustomMergeStrategy getStrategy() {
      return strategy;
    }

    /** Returns the mergeBuffer. */
    public ByteArrayOutputStream getMergeBuffer() {
      return mergeBuffer;
    }
  }

  private final class FilterCallback implements StrategyCallback {
    private String filename;
    private final AtomicBoolean called = new AtomicBoolean();

    public void resetForFile(String filename) {
      this.filename = filename;
      this.called.set(false);
    }

    @Override public void skip() throws IOException {
      checkCall();
      actions.put(filename, new EntryAction(ActionType.SKIP));
    }

    @Override public void copy(Date date) throws IOException {
      checkCall();
      actions.put(filename, new EntryAction(ActionType.COPY, date, null, null, null));
    }

    @Override public void rename(String newName, Date date) throws IOException {
      checkCall();
      actions.put(filename, new EntryAction(ActionType.RENAME, date, newName, null, null));
    }

    @Override public void customMerge(Date date, CustomMergeStrategy strategy) throws IOException {
      checkCall();
      actions.put(filename, new EntryAction(ActionType.MERGE, date, null, strategy,
          new ByteArrayOutputStream()));
    }

    private void checkCall() {
      checkState(called.compareAndSet(false, true), "The callback was already called once.");
    }
  }

  /** Returns a {@link Deflater} for performing ZIP compression. */
  private static Deflater getDeflater() {
    return new Deflater(Deflater.DEFAULT_COMPRESSION, true);
  }

  /** Returns a {@link Inflater} for performing ZIP decompression. */
  private static Inflater getInflater() {
    return new Inflater(true);
  }

  /** Copies all data from the input stream to the output stream. */
  private static long copyStream(InputStream from, OutputStream to) throws IOException {
    byte[] buf = new byte[0x1000];
    long total = 0;
    int r;
    while ((r = from.read(buf)) != -1) {
      to.write(buf, 0, r);
      total += r;
    }
    return total;
  }

  private final OutputMode mode;
  private final ZipEntryFilter entryFilter;
  private final FilterCallback callback;
  private final ZipWriter out;

  private final Map<String, ZipFileEntry> entries;
  private final Map<String, EntryAction> actions;

  /**
   * Creates a {@link ZipCombiner} for combining ZIP files using the specified {@link OutputMode},
   * {@link ZipEntryFilter}, and destination {@link OutputStream}.
   *
   * @param mode the compression preference for the output ZIP file
   * @param entryFilter the filter to use when adding ZIP files to the combined output
   * @param out the {@link OutputStream} for writing the combined ZIP file
   */
  public ZipCombiner(OutputMode mode, ZipEntryFilter entryFilter, OutputStream out) {
    this.mode = mode;
    this.entryFilter = entryFilter;
    this.callback = new FilterCallback();
    this.out = new ZipWriter(new BufferedOutputStream(out), UTF_8);
    this.entries = new HashMap<>();
    this.actions = new HashMap<>();
  }

  /**
   * Creates a {@link ZipCombiner} for combining ZIP files using the specified
   * {@link ZipEntryFilter}, and destination {@link OutputStream}. Uses the DONT_CARE
   * {@link OutputMode}.
   *
   * @param entryFilter the filter to use when adding ZIP files to the combined output
   * @param out the {@link OutputStream} for writing the combined ZIP file
   */
  public ZipCombiner(ZipEntryFilter entryFilter, OutputStream out) {
    this(OutputMode.DONT_CARE, entryFilter, out);
  }

  /**
   * Creates a {@link ZipCombiner} for combining ZIP files using the specified {@link OutputMode},
   * and destination {@link OutputStream}. Uses a {@link CopyEntryFilter} as the
   * {@link ZipEntryFilter}.
   *
   * @param mode the compression preference for the output ZIP file
   * @param out the {@link OutputStream} for writing the combined ZIP file
   */
  public ZipCombiner(OutputMode mode, OutputStream out) {
    this(mode, new CopyEntryFilter(), out);
  }

  /**
   * Creates a {@link ZipCombiner} for combining ZIP files using the specified destination
   * {@link OutputStream}. Uses the DONT_CARE {@link OutputMode} and a {@link CopyEntryFilter} as
   * the {@link ZipEntryFilter}.
   *
   * @param out the {@link OutputStream} for writing the combined ZIP file
   */
  public ZipCombiner(OutputStream out) {
    this(OutputMode.DONT_CARE, new CopyEntryFilter(), out);
  }

  /**
   * Adds a directory entry to the combined ZIP file using the specified filename, date, and extra
   * data.
   *
   * @param filename the name of the directory to create
   * @param date the modified time to assign to the directory
   * @param extra the extra field data to add to the directory entry
   * @throws IOException if there is an error writing the directory entry
   */
  public void addDirectory(String filename, Date date, ExtraData[] extra) throws IOException {
    checkArgument(filename.endsWith("/"), "Directory names must end with a /");
    checkState(!entries.containsKey(filename),
        "Zip already contains a directory named %s", filename);

    ZipFileEntry entry = new ZipFileEntry(filename);
    entry.setMethod(Compression.STORED);
    entry.setCrc(0);
    entry.setSize(0);
    entry.setCompressedSize(0);
    entry.setTime(date != null ? date.getTime() : new Date().getTime());
    entry.setExtra(new ExtraDataList(extra));
    out.putNextEntry(entry);
    out.closeEntry();
    entries.put(filename, entry);
  }

  /**
   * Adds a file with the specified name and date to the combined ZIP file.
   *
   * @param filename the name of the file to create
   * @param date the modified time to assign to the file
   * @param in the {@link InputStream} containing the file data
   * @throws IOException if there is an error writing the file entry
   * @throws IllegalArgumentException if the combined ZIP file already contains a file of the same
   *     name.
   */
  public void addFile(String filename, Date date, InputStream in) throws IOException {
    ZipFileEntry entry = new ZipFileEntry(filename);
    entry.setTime(date != null ? date.getTime() : new Date().getTime());
    addFile(entry, in);
  }

  /**
   * Adds a file with attributes specified by the {@link ZipFileEntry} to the combined ZIP file.
   *
   * @param entry the {@link ZipFileEntry} containing the entry meta-data
   * @param in the {@link InputStream} containing the file data
   * @throws IOException if there is an error writing the file entry
   * @throws IllegalArgumentException if the combined ZIP file already contains a file of the same
   *     name.
   */
  public void addFile(ZipFileEntry entry, InputStream in) throws IOException {
    checkNotNull(entry, "Zip entry must not be null.");
    checkNotNull(in, "Input stream must not be null.");
    checkArgument(!entries.containsKey(entry.getName()), "Zip already contains a file named '%s'.",
        entry.getName());

    ByteArrayOutputStream uncompressed = new ByteArrayOutputStream();
    copyStream(in, uncompressed);

    writeEntryFromBuffer(new ZipFileEntry(entry), uncompressed.toByteArray());
  }

  /**
   * Adds the contents of a ZIP file to the combined ZIP file using the specified
   * {@link ZipEntryFilter} to determine the appropriate action for each file. 
   *
   * @param zipFile the ZIP file to add to the combined ZIP file
   * @throws IOException if there is an error reading the ZIP file or writing entries to the
   *     combined ZIP file
   */
  public void addZip(File zipFile) throws IOException {
    try (ZipReader zip = new ZipReader(zipFile)) {
      for (ZipFileEntry entry : zip.entries()) {
        String filename = entry.getName();
        EntryAction action = getAction(filename);
        switch (action.getType()) {
          case SKIP:
            break;
          case COPY:
          case RENAME:
            writeEntry(zip, entry, action);
            break;
          case MERGE:
            entries.put(filename, null);
            InputStream in = zip.getRawInputStream(entry);
            if (entry.getMethod() == Compression.DEFLATED) {
              in = new InflaterInputStream(in, getInflater(), INFLATER_BUFFER_BYTES);
            }
            action.getStrategy().merge(in, action.getMergeBuffer());
            break;
        }
      }
    }
  }

  /** Returns the action to take for a file of the given filename. */
  private EntryAction getAction(String filename) throws IOException {
    // If this filename has not been encountered before (no entry for filename) or this filename
    // has been renamed (RENAME entry for filename), the desired action should be recomputed.
    if (!actions.containsKey(filename) || actions.get(filename).getType() == ActionType.RENAME) {
      callback.resetForFile(filename);
      entryFilter.accept(filename, callback);
    }
    checkState(actions.containsKey(filename),
        "Action for file '%s' should have been set by ZipEntryFilter.", filename);

    EntryAction action = actions.get(filename);
    // Only copy if this is the first instance of filename.
    if (action.getType() == ActionType.COPY && entries.containsKey(filename)) {
      action = new EntryAction(ActionType.SKIP, action);
      actions.put(filename, action);
    }
    // Only rename if there is not already an entry with filename or filename's action is SKIP.
    if (action.getType() == ActionType.RENAME) {
      if (actions.containsKey(action.getNewName())
          && actions.get(action.getNewName()).getType() == ActionType.SKIP) {
        action = new EntryAction(ActionType.SKIP, action);
      }
      if (entries.containsKey(action.getNewName())) {
        action = new EntryAction(ActionType.SKIP, action);
      }
    }
    return action;
  }

  /** Writes an entry with the given name, date and external file attributes from the buffer. */
  private void writeEntryFromBuffer(ZipFileEntry entry, byte[] uncompressed) throws IOException {
    CRC32 crc = new CRC32();
    crc.update(uncompressed);

    entry.setCrc(crc.getValue());
    entry.setSize(uncompressed.length);
    if (mode == OutputMode.FORCE_STORED) {
      entry.setMethod(Compression.STORED);
      entry.setCompressedSize(uncompressed.length);
      writeEntry(entry, new ByteArrayInputStream(uncompressed));
    } else {
      ByteArrayOutputStream compressed = new ByteArrayOutputStream();
      copyStream(
          new DeflaterInputStream(
              new ByteArrayInputStream(uncompressed), getDeflater(), INFLATER_BUFFER_BYTES),
          compressed);
      entry.setMethod(Compression.DEFLATED);
      entry.setCompressedSize(compressed.size());
      writeEntry(entry, new ByteArrayInputStream(compressed.toByteArray()));
    }
  }

  /**
   * Writes an entry from the specified source {@link ZipReader} and {@link ZipFileEntry} using the
   * specified {@link EntryAction}.
   * 
   *  <p>Writes the output entry from the input entry performing inflation or deflation as needed
   *  and applies any values from the {@link EntryAction} as needed.
   */
  private void writeEntry(ZipReader zip, ZipFileEntry entry, EntryAction action)
      throws IOException {
    checkArgument(action.getType() != ActionType.SKIP,
        "Cannot write a zip entry whose action is of type SKIP.");

    ZipFileEntry outEntry = new ZipFileEntry(entry);
    if (action.getType() == ActionType.RENAME) {
      checkNotNull(action.getNewName(),
          "ZipEntryFilter actions of type RENAME must not have a null filename.");
      outEntry.setName(action.getNewName());
    }

    if (action.getDate() != null) {
      outEntry.setTime(action.getDate().getTime());
    }

    InputStream data;
    if (mode == OutputMode.FORCE_DEFLATE && entry.getMethod() != Compression.DEFLATED) {
      // The output mode is deflate, but the entry compression is not. Create a deflater stream
      // from the raw file data and deflate to a temporary byte array to determine the deflated
      // size. Then use this byte array as the input stream for writing the entry.
      ByteArrayOutputStream tmp = new ByteArrayOutputStream();
      copyStream(
          new DeflaterInputStream(
              zip.getRawInputStream(entry), getDeflater(), INFLATER_BUFFER_BYTES),
          tmp);
      data = new ByteArrayInputStream(tmp.toByteArray());
      outEntry.setMethod(Compression.DEFLATED);
      outEntry.setCompressedSize(tmp.size());
    } else if (mode == OutputMode.FORCE_STORED && entry.getMethod() != Compression.STORED) {
      // The output mode is stored, but the entry compression is not; create an inflater stream
      // from the raw file data.
      data =
          new InflaterInputStream(
              zip.getRawInputStream(entry), getInflater(), INFLATER_BUFFER_BYTES);
      outEntry.setMethod(Compression.STORED);
      outEntry.setCompressedSize(entry.getSize());
    } else {
      // Entry compression agrees with output mode; use the raw file data as is.
      data = zip.getRawInputStream(entry);
    }
    writeEntry(outEntry, data);
  }

  /**
   * Writes the specified {@link ZipFileEntry} using the data from the given {@link InputStream}.
   */
  private void writeEntry(ZipFileEntry entry, InputStream data) throws IOException {
    out.putNextEntry(entry);
    copyStream(data, out);
    out.closeEntry();
    entries.put(entry.getName(), entry);
  }

  /**
   * Writes any remaining output data to the output stream and also creates the merged entries by
   * calling the {@link CustomMergeStrategy} implementations given back from the
   * {@link ZipEntryFilter}.
   *
   * @throws IOException if the output stream or the filter throws an IOException
   * @throws IllegalStateException if this method was already called earlier
   */
  public void finish() throws IOException {
    for (Map.Entry<String, EntryAction> entry : actions.entrySet()) {
      String filename = entry.getKey();
      EntryAction action = entry.getValue();
      if (action.getType() == ActionType.MERGE) {
        ByteArrayOutputStream uncompressed = action.getMergeBuffer();
        action.getStrategy().finish(uncompressed);
        if (uncompressed.size() == 0 && action.getStrategy().skipEmpty()) {
          continue;
        }

        ZipFileEntry e = new ZipFileEntry(filename);
        e.setTime(action.getDate() != null ? action.getDate().getTime() : new Date().getTime());
        writeEntryFromBuffer(e, uncompressed.toByteArray());
      }
    }
    out.finish();
  }

  /**
   * Writes any remaining output data to the output stream and closes it.
   *
   * @throws IOException if the output stream or the filter throws an IOException
   */
  @Override public void close() throws IOException {
    finish();
    out.close();
  }

  /** Ensures the truth of an expression involving one or more parameters to the calling method. */
  private static void checkArgument(boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object... errorMessageArgs) {
    if (!expression) {
      throw new IllegalArgumentException(String.format(errorMessageTemplate, errorMessageArgs));
    }
  }

  /** Ensures that an object reference passed as a parameter to the calling method is not null. */
  public static <T> T checkNotNull(T reference,
      @Nullable String errorMessageTemplate,
      @Nullable Object... errorMessageArgs) {
    if (reference == null) {
      // If either of these parameters is null, the right thing happens anyway
      throw new NullPointerException(String.format(errorMessageTemplate, errorMessageArgs));
    }
    return reference;
  }

  /** Ensures the truth of an expression involving state. */
  private static void checkState(boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object... errorMessageArgs) {
    if (!expression) {
      throw new IllegalStateException(String.format(errorMessageTemplate, errorMessageArgs));
    }
  }
}
