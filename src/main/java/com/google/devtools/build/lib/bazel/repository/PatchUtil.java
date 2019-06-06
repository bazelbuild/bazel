// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;


import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.vfs.Path;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import difflib.Chunk;
import difflib.Delta;
import difflib.DeltaComparator;
import difflib.DiffUtils;
import difflib.Patch;
import difflib.PatchFailedException;

public class PatchUtil {

  private static Pattern chunkHeaderRe =
      Pattern.compile("^@@\\s+-(?:(\\d+)(?:,(\\d+))?)\\s+\\+(?:(\\d+)(?:,(\\d+))?)\\s+@@$");

  public enum Result {
    COMPLETE,  // The entire chunk is read
    CONTINUE,  // Should continue reading the chunk
    ERROR,     // The chunk body doesn't match the chunk header's size description
  }

  private static class ChunkHeader {
    private int oldSize;
    private int newSize;

    public Result check(int oldLineCnt, int newLineCnt) {
      if (oldLineCnt == oldSize && newLineCnt == newSize) {
        return Result.COMPLETE;
      }
      if (oldLineCnt <= oldSize && newLineCnt <= newSize) {
        return Result.CONTINUE;
      }
      return Result.ERROR;
    }

    ChunkHeader(String header) throws PatchFailedException {
      Matcher m = chunkHeaderRe.matcher(header);
      if (m.find()) {
        oldSize = Integer.parseInt(m.group(2));
        newSize = Integer.parseInt(m.group(4));
      } else {
        throw new PatchFailedException("Wrong chunk header: " + header);
      }
    }
  }

  /**
   * Sometimes the line number in patch file is not completely correct, but we might still be able
   * to find a content match with an offset.
   */
  private static class OffsetPatch {
    private List<Delta<String>> deltas;

    public OffsetPatch(Patch<String> patch) {
      this.deltas = patch.getDeltas();
    }

    public List<String> applyTo(List<String> target) throws PatchFailedException {
      List<String> result = new ArrayList<>(target);
      this.deltas.sort(DeltaComparator.INSTANCE);
      ListIterator<Delta<String>> it = this.deltas.listIterator(this.deltas.size());

      while(it.hasPrevious()) {
        Delta<String> delta = it.previous();
        applyTo(delta, result);
      }

      return result;
    }

    /**
     * This function first tries to apply the Delta without any offset, if that fails, then
     * it tries to apply the Delta with an offset, starting from 1, up to the total lines in
     * the original content. For every offset, we try both forwards and backwards.
     */
    private void applyTo(Delta<String> delta, List<String> result) throws PatchFailedException {
      PatchFailedException e = applyDelta(delta, result);
      if (e == null) {
        return;
      }

      Chunk<String> original = delta.getOriginal();
      Chunk<String> revised = delta.getRevised();
      int[] direction = {1, -1};
      int maxOffset = result.size();
      for (int i = 1; i < maxOffset; i++) {
        for (int j = 0; j < 2; j++) {
          int offset = i * direction[j];
          if (offset + original.getPosition() < 0 || offset + revised.getPosition() < 0) {
            continue;
          }
          delta.setOriginal(
              new Chunk<>(original.getPosition() + offset, original.getLines()));
          delta.setRevised(
              new Chunk<>(revised.getPosition() + offset, revised.getLines()));
          PatchFailedException exception = applyDelta(delta, result);
          if (exception == null) {
            return;
          }
        }
      }

      throw e;
    }

    private PatchFailedException applyDelta(Delta<String> delta, List<String> result) {
      try {
        delta.applyTo(result);
        return null;
      } catch (PatchFailedException e) {
        return e;
      }
    }
  }

  private enum LineType {
    OLD_FILE,
    NEW_FILE,
    CHUNK_HEAD,
    CHUNK_ADD,
    CHUNK_DEL,
    CHUNK_EQL,
    UNKNOWN
  }

  private static LineType getLineType(String line, boolean isReadingChunk) {
    if (!isReadingChunk && line.startsWith("---")) {
      return LineType.OLD_FILE;
    }
    if (!isReadingChunk && line.startsWith("+++")) {
      return LineType.NEW_FILE;
    }
    if (line.startsWith("@@") && line.lastIndexOf("@@") != 0) {
      int pos = line.indexOf("@@", 2);
      Matcher m = chunkHeaderRe.matcher(line.substring(0, pos + 2));
      if (m.find()) {
        return LineType.CHUNK_HEAD;
      } else {
        return LineType.UNKNOWN;
      }
    }
    if (isReadingChunk && line.startsWith("+")) {
      return LineType.CHUNK_ADD;
    }
    if (isReadingChunk && line.startsWith("-")) {
      return LineType.CHUNK_DEL;
    }
    if (isReadingChunk && (line.startsWith(" ") || line.isEmpty())) {
      return LineType.CHUNK_EQL;
    }
    return LineType.UNKNOWN;
  }

  /**
   * If file is not null and the file exists, return the file content as a list for String.
   * Otherwise, return an empty list.
   */
  @VisibleForTesting
  public static List<String> readFile(Path file) throws IOException {
    List<String> content = new ArrayList<>();
    if (file != null && file.exists()) {
      BufferedReader reader = new BufferedReader(new InputStreamReader(file.getInputStream()));
      String line;
      while ((line = reader.readLine()) != null) {
        content.add(line);
      }
      reader.close();
    }
    return content;
  }

  private static void writeFile(Path file, List<String> content) throws IOException {
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(file.getOutputStream()));
    for (String line : content) {
      writer.write(line);
      writer.write("\n");
    }
    writer.close();
  }

  private static void applyPatchToFile(List<String> patchContent, Path oldFile, Path newFile)
      throws IOException, PatchFailedException {
    // The oldFile could be <newFile>.orig, but the .orig may not exist, so in this case,
    // we consider oldFile is the same as newFile.
    if (oldFile != null && newFile != null) {
      if (oldFile.getBaseName().equals(newFile.getBaseName().concat(".orig"))) {
        oldFile = newFile;
      }
    }

    List<String> oldContent = readFile(oldFile);
    Patch<String> patch = DiffUtils.parseUnifiedDiff(patchContent);
    List<String> newContent = new OffsetPatch(patch).applyTo(oldContent);

    if (newContent.isEmpty() && newFile == null) {
      oldFile.delete();
      return;
    }

    if (newFile != null) {
      writeFile(newFile, newContent);
    }
  }

  /**
   * Strip a number of leading components from a path
   * @param path the original path
   * @param strip the number of leading components to strip
   * @return The stripped path
   */
  private static String stripPath(String path, int strip) {
    int pos = 0;
    while (pos < path.length() && strip > 0) {
      if (path.charAt(pos) == '/') {
        strip--;
      }
      pos++;
    }
    return path.substring(pos);
  }

  private static Path getFilePath(String line, int strip, Path outputDirectory) {
    // The line could look like:
    // --- a/foo/bar.txt   2019-05-27 17:19:37.054593200 +0200
    // +++ b/foo/bar.txt   2019-05-27 17:19:37.054593200 +0200
    // If strip is 1, we want get the file path as <outputDirectory>/foo/bar.txt
    String file = null;
    line = line.split("\t")[0];
    if (line.length() > 4) {
      if (line.substring(4).trim().equals("/dev/null")) {
        return null;
      }

      file = stripPath(line.substring(4), strip);
      if (!file.isEmpty()) {
        return outputDirectory.getRelative(file);
      }
    }
    return null;
  }

  /**
   * Apply a patch file under a directory
   * @param patchFile the patch file to apply
   * @param strip the number of leading components to strip from file path in the patch file
   * @param outputDirectory the repository directory to apply the patch file
   */
  public static void apply(Path patchFile, int strip, Path outputDirectory)
      throws IOException, PatchFailedException {
    List<String> patch = readFile(patchFile);
    // Adding an extra line to make sure last chunk also gets applied.
    patch.add("$");

    boolean isReadingChunk = false;
    List<String> patchContent = new ArrayList<>();
    ChunkHeader header = null;
    Path oldFile = null;
    Path newFile = null;
    int oldLineCount = 0;
    int newLineCount = 0;
    Result result;

    for (int i = 0; i < patch.size(); i++) {
      String line = patch.get(i);
      switch (getLineType(line, isReadingChunk)) {
        case OLD_FILE:
          patchContent.add(line);
          oldFile = getFilePath(line, strip, outputDirectory);
          break;
        case NEW_FILE:
          patchContent.add(line);
          newFile = getFilePath(line, strip, outputDirectory);
          break;
        case CHUNK_HEAD:
          int pos = line.indexOf("@@", 2);
          String headerStr = line.substring(0, pos + 2);
          patchContent.add(headerStr);
          header = new ChunkHeader(headerStr);
          oldLineCount = 0;
          newLineCount = 0;
          isReadingChunk = true;
          break;
        case CHUNK_ADD:
          newLineCount++;
          patchContent.add(line);
          result = header.check(oldLineCount, newLineCount);
          if (result == Result.COMPLETE) {
            isReadingChunk = false;
          } else if (result == Result.ERROR) {
            throw new PatchFailedException(
                "Wrong chunk detected near line " + (i + 1) + ": " + line);
          }
          break;
        case CHUNK_DEL:
          oldLineCount++;
          patchContent.add(line);
          result = header.check(oldLineCount, newLineCount);
          if (result == Result.COMPLETE) {
            isReadingChunk = false;
          } else if (result == Result.ERROR) {
            throw new PatchFailedException(
                "Wrong chunk detected near line " + (i + 1) + ": " + line);
          }
          break;
        case CHUNK_EQL:
          oldLineCount++;
          newLineCount++;
          patchContent.add(line);
          result = header.check(oldLineCount, newLineCount);
          if (result == Result.COMPLETE) {
            isReadingChunk = false;
          } else if (result == Result.ERROR) {
            throw new PatchFailedException(
                "Wrong chunk detected near line " + (i + 1) + ": " + line);
          }
          break;
        case UNKNOWN:
          if (!patchContent.isEmpty() && (oldFile != null || newFile != null)) {
            result = header.check(oldLineCount, newLineCount);
            // result will never be Result.Error here because it would have been throw in previous
            // line already.
            if (result == Result.CONTINUE) {
              throw new PatchFailedException("Expecting more chunk line at line " + (i + 1));
            }
            applyPatchToFile(patchContent, oldFile, newFile);
          }
          patchContent.clear();
          oldFile = null;
          newFile = null;
          oldLineCount = 0;
          newLineCount = 0;
          isReadingChunk = false;
          break;
      }
    }
  }
}
