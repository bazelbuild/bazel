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

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Command-line entry point for the dex reducer utility. This utility extracts .dex files
 * from one or more archives, and packaging them in a single output archive, renaming entries
 * to: classes.dex, classes2.dex, ...
 *
 * <p>This utility is intended used to consolidate the result of compiling the output produced
 * by the dex mapper utility.</p>
 */
public class DexReducer implements EntryHandler {
  private static final String SUFFIX = ".dex";
  private static final String BASENAME = "classes";
  private ZipOut out;
  private int count = 0;
  private String outFile;
  private List<String> paths;

  DexReducer() {
    outFile = null;
    paths = new ArrayList<>();
  }

  /**
   * Command-line entry point.
   * @param args
   */
  public static void main(String[] args) {
    try {
     DexReducer dexDexReducer = new DexReducer();
     dexDexReducer.parseArguments(args);
     dexDexReducer.run();
    } catch (Exception ex) {
      System.err.println("DexReducer threw exception: " + ex.getMessage());
      System.exit(1);
    }
  }

  private void parseArguments(String[] args) {
    DexReducerOptions options =
        Options.parseAndExitUponError(DexReducerOptions.class, /*allowResidue=*/ true, args)
            .getOptions();
    paths = options.inputZips;
    outFile = options.outputZip;
  }

  private void run() throws IOException {
    out = new ZipOut(new FileOutputStream(outFile, false).getChannel(), outFile);
    for (String filename : paths) {
      ZipIn in = new ZipIn(new FileInputStream(filename).getChannel(), filename);
      in.scanEntries(this);
    }
    out.close();
  }

  @Override
  public void handle(ZipIn in, LocalFileHeader header, DirectoryEntry dirEntry, ByteBuffer data)
      throws IOException {
    String path = header.getFilename();
    if (!path.endsWith(".dex")) {
      return;
    }
    count++;
    String filename = BASENAME + (count == 1 ? "" : Integer.toString(count)) + SUFFIX;
    String comment = dirEntry.getComment();
    byte[] extra = dirEntry.getExtraData();
    out.nextEntry(dirEntry.clone(filename, extra, comment).set(CENTIM, DosTime.EPOCH.time));
    out.write(header.clone(filename, extra).set(LOCTIM, DosTime.EPOCH.time));
    out.write(data);
    if ((header.get(LOCFLG) & LocalFileHeader.SIZE_MASKED_FLAG) != 0) {
      DataDescriptor desc = DataDescriptor.allocate()
          .set(EXTCRC, dirEntry.get(CENCRC))
          .set(EXTSIZ, dirEntry.get(CENSIZ))
          .set(EXTLEN, dirEntry.get(CENLEN));
      out.write(desc);
    }
  }

  /** Commandline options. */
  public static class DexReducerOptions extends OptionsBase {
    @Option(
      name = "input_zip",
      defaultValue = "null",
      category = "input",
      allowMultiple = true,
      abbrev = 'i',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Input zip file containing entries to collect and enumerate."
    )
    public List<String> inputZips;

    @Option(
      name = "output_zip",
      defaultValue = "null",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      abbrev = 'o',
      help = "Output zip file, containing enumerated entries."
    )
    public String outputZip;
  }
}
