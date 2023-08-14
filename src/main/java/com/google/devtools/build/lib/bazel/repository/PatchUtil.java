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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.github.difflib.UnifiedDiffUtils;
import com.github.difflib.patch.AbstractDelta;
import com.github.difflib.patch.ChangeDelta;
import com.github.difflib.patch.Chunk;
import com.github.difflib.patch.DeleteDelta;
import com.github.difflib.patch.InsertDelta;
import com.github.difflib.patch.Patch;
import com.github.difflib.patch.PatchFailedException;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** Implementation of native patch. */
public class PatchUtil {

  private static final Pattern CHUNK_HEADER_RE =
      Pattern.compile("^@@\\s+-(?:(\\d+)(?:,(\\d+))?)\\s+\\+(?:(\\d+)(?:,(\\d+))?)\\s+@@$");

  /** The possible results of ChunkHeader.check. */
  public enum Result {
    COMPLETE, // The entire chunk is read
    CONTINUE, // Should continue reading the chunk
    ERROR, // The chunk body doesn't match the chunk header's size description
  }

  private static class ChunkHeader {
    private final int oldSize;
    private final int newSize;

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
      Matcher m = CHUNK_HEADER_RE.matcher(header);
      if (m.find()) {
        String size;
        size = m.group(2);
        oldSize = (size == null) ? 1 : Integer.parseInt(size);
        size = m.group(4);
        newSize = (size == null) ? 1 : Integer.parseInt(size);
      } else {
        throw new PatchFailedException("Wrong chunk header: " + header);
      }
    }
  }

  // Sometimes the line number in patch file is not completely correct, but we might still be able
  // to find a content match with an offset.
  private static List<String> applyOffsetPatchTo(Patch<String> patch, ImmutableList<String> target)
      throws PatchFailedException {
    List<AbstractDelta<String>> deltas = patch.getDeltas();
    List<String> result = new ArrayList<>(target);
    for (AbstractDelta<String> item : Lists.reverse(deltas)) {
      AbstractDelta<String> delta = item;
      applyTo(delta, result);
    }

    return result;
  }

  /**
   * This function first tries to apply the Delta without any offset, if that fails, then it tries
   * to apply the Delta with an offset, starting from 1, up to the total lines in the original
   * content. For every offset, we try both forwards and backwards.
   */
  private static void applyTo(AbstractDelta<String> delta, List<String> result)
      throws PatchFailedException {
    PatchFailedException e = applyDelta(delta, result);
    if (e == null) {
      return;
    }

    Chunk<String> original = delta.getSource();
    Chunk<String> revised = delta.getTarget();
    int[] direction = {1, -1};
    int maxOffset = result.size();
    for (int i = 1; i < maxOffset; i++) {
      for (int j = 0; j < 2; j++) {
        int offset = i * direction[j];
        if (offset + original.getPosition() < 0 || offset + revised.getPosition() < 0) {
          continue;
        }
        Chunk<String> source = new Chunk<>(original.getPosition() + offset, original.getLines());
        Chunk<String> target = new Chunk<>(revised.getPosition() + offset, revised.getLines());
        AbstractDelta<String> newDelta = null;
        switch (delta.getType()) {
          case CHANGE:
            newDelta = new ChangeDelta<>(source, target);
            break;
          case INSERT:
            newDelta = new InsertDelta<>(source, target);
            break;
          case DELETE:
            newDelta = new DeleteDelta<>(source, target);
            break;
          case EQUAL:
        }
        PatchFailedException exception = null;
        if (newDelta != null) {
          exception = applyDelta(newDelta, result);
        }
        if (exception == null) {
          return;
        }
      }
    }

    throw e;
  }

  @Nullable
  private static PatchFailedException applyDelta(AbstractDelta<String> delta, List<String> result) {
    try {
      delta.applyTo(result);
      return null;
    } catch (PatchFailedException e) {
      String msg =
          String.join(
              "\n",
              "**Original Position**: " + (delta.getSource().getPosition() + 1) + "\n",
              "**Original Content**:",
              String.join("\n", delta.getSource().getLines()) + "\n",
              "**Revised Content**:",
              String.join("\n", delta.getTarget().getLines()) + "\n");
      return new PatchFailedException(e.getMessage() + "\n" + msg);
    }
  }

  private enum LineType {
    OLD_FILE,
    NEW_FILE,
    CHUNK_HEAD,
    CHUNK_ADD,
    CHUNK_DEL,
    CHUNK_EQL,
    GIT_HEADER,
    RENAME_FROM,
    RENAME_TO,
    NEW_MODE,
    NEW_FILE_MODE,
    OTHER_GIT_LINE,
    UNKNOWN
  }

  private static final String[] GIT_LINE_PREFIXES = {
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "copy from ",
    "copy to ",
    "rename old ",
    "rename new ",
    "similarity index ",
    "dissimilarity index ",
    "index "
  };

  private static LineType getLineType(String line, boolean isReadingChunk, boolean isGitDiff) {
    if (isReadingChunk) {
      if (line.startsWith("+")) {
        return LineType.CHUNK_ADD;
      }
      if (line.startsWith("-")) {
        return LineType.CHUNK_DEL;
      }
      if (line.startsWith(" ") || line.isEmpty()) {
        return LineType.CHUNK_EQL;
      }
    } else {
      if (line.startsWith("--- ")) {
        return LineType.OLD_FILE;
      }
      if (line.startsWith("+++ ")) {
        return LineType.NEW_FILE;
      }
      if (line.startsWith("diff --git ")) {
        return LineType.GIT_HEADER;
      }
      if (isGitDiff) {
        // Only recognize the following when we saw "diff --git " before.
        if (line.startsWith("rename from ")) {
          return LineType.RENAME_FROM;
        }
        if (line.startsWith("rename to ")) {
          return LineType.RENAME_TO;
        }
        if (line.startsWith("new mode ")) {
          return LineType.NEW_MODE;
        }
        if (line.startsWith("new file mode ")) {
          return LineType.NEW_FILE_MODE;
        }
        for (String prefix : GIT_LINE_PREFIXES) {
          if (line.startsWith(prefix)) {
            return LineType.OTHER_GIT_LINE;
          }
        }
      }
    }
    if (line.startsWith("@@") && line.lastIndexOf("@@") != 0) {
      int pos = line.indexOf("@@", 2);
      Matcher m = CHUNK_HEADER_RE.matcher(line.substring(0, pos + 2));
      if (m.find()) {
        return LineType.CHUNK_HEAD;
      }
    }
    return LineType.UNKNOWN;
  }

  private static ImmutableList<String> readFile(Path file) throws IOException {
    return FileSystemUtils.readLines(file, UTF_8);
  }

  private static void writeFile(Path file, List<String> content) throws IOException {
    FileSystemUtils.writeLinesAs(file, UTF_8, content);
  }

  private static boolean getReadPermission(int permission) {
    // Parse read permission from posix file permission notation
    return (permission & 4) == 4;
  }

  private static boolean getWritePermission(int permission) {
    // Parse write permission from posix file permission notation
    return (permission & 2) == 2;
  }

  private static boolean getExecutablePermission(int permission) {
    // Parse executable permission from posix file permission notation
    return (permission & 1) == 1;
  }

  private static int getFilePermissionValue(Path file) throws IOException {
    return (file.isReadable() ? 4 : 0)
        + (file.isWritable() ? 2 : 0)
        + (file.isExecutable() ? 1 : 0);
  }

  private static void setFilePermission(Path file, int permission) throws IOException {
    file.setReadable(getReadPermission(permission));
    file.setWritable(getWritePermission(permission));
    file.setExecutable(getExecutablePermission(permission));
  }

  private static void applyPatchToFile(
      Patch<String> patch, Path oldFile, Path newFile, boolean isRenaming, int filePermission)
      throws IOException, PatchFailedException {
    // The file we should read oldContent from.
    Path inputFile = null;
    if (oldFile != null && oldFile.exists()) {
      inputFile = oldFile;
    } else if (newFile != null && newFile.exists()) {
      inputFile = newFile;
    }

    ImmutableList<String> oldContent;
    if (inputFile == null) {
      oldContent = ImmutableList.of();
    } else {
      oldContent = readFile(inputFile);
      // Preserve old file permission if no explicit permission is set.
      if (filePermission == -1) {
        filePermission = getFilePermissionValue(inputFile);
      }
    }

    List<String> newContent;
    try {
      newContent = applyOffsetPatchTo(patch, oldContent);
    } catch (PatchFailedException e) {
      throw new PatchFailedException(
          String.format("in patch applied to %s: %s", oldFile, e.getMessage()));
    }

    // The file we should write newContent to.
    Path outputFile;
    if (oldFile != null && oldFile.exists() && !isRenaming) {
      outputFile = oldFile;
    } else {
      outputFile = newFile;
    }

    // The old file should always change, therefore we can just delete the original file.
    // If the output file name is the same as the old file, we'll just recreate it later.
    if (oldFile != null) {
      oldFile.delete();
    }

    // Does this patch look like deleting a file.
    boolean isDeleteFile = newFile == null && newContent.isEmpty();

    if (outputFile != null && !isDeleteFile) {
      writeFile(outputFile, newContent);
      if (filePermission != -1) {
        setFilePermission(outputFile, filePermission);
      }
    }
  }

  /**
   * Strip a number of leading components from a path
   *
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

  /**
   * Extract the file path from a patch line starting with "--- " or "+++ " Returns null if the path
   * is /dev/null, otherwise returns the extracted path if succeeded or throw an exception if
   * failed.
   */
  @Nullable
  private static String extractPath(String line, int strip, int loc) throws PatchFailedException {
    // The line could look like:
    // --- a/foo/bar.txt   2019-05-27 17:19:37.054593200 +0200
    // +++ b/foo/bar.txt   2019-05-27 17:19:37.054593200 +0200
    // If strip is 1, we want extract the file path as foo/bar.txt
    Preconditions.checkArgument(line.startsWith("+++ ") || line.startsWith("--- "));
    line = Iterables.get(Splitter.on('\t').split(line), 0);
    if (line.length() > 4) {
      String path = line.substring(4).trim();
      if (path.equals("/dev/null")) {
        return null;
      }
      path = stripPath(path, strip);
      if (!path.isEmpty()) {
        return path;
      }
    }
    throw new PatchFailedException(
        String.format(
            "Cannot determine file name with strip = %d at line %d:\n%s", strip, loc, line));
  }

  @Nullable
  private static Path getFilePath(String path, Path outputDirectory, int loc)
      throws PatchFailedException {
    if (path == null) {
      return null;
    }
    Path filePath = outputDirectory.getRelative(path);
    if (!filePath.startsWith(outputDirectory)) {
      throw new PatchFailedException(
          String.format(
              "Cannot patch file outside of external repository (%s), file path = \"%s\" at line"
                  + " %d",
              outputDirectory.getPathString(), path, loc));
    }
    return filePath;
  }

  private static void checkPatchContentIsComplete(
      List<String> patchContent, ChunkHeader header, int oldLineCount, int newLineCount, int loc)
      throws PatchFailedException {
    // If the patchContent is not empty, it should have correct format.
    if (!patchContent.isEmpty()) {
      if (patchContent.size() < 2
          || !patchContent.get(0).startsWith("---")
          || !patchContent.get(1).startsWith("+++")) {
        throw new PatchFailedException(
            String.format(
                "The patch content must start with ---/+++ prelude lines at line %d.", loc));
      }
      if (header == null) {
        throw new PatchFailedException(
            String.format(
                "Looks like a unified diff at line %d, but no patch chunk was found.", loc));
      }
      Result result = header.check(oldLineCount, newLineCount);
      // result will never be Result.Error here because it would have been throw in previous
      // line already.
      if (result == Result.CONTINUE) {
        throw new PatchFailedException(
            String.format("Expecting more chunk line at line %d", loc + patchContent.size()));
      }
    }
  }

  private static void checkFilesStatusForRenaming(
      Path oldFile, Path newFile, String oldFileStr, String newFileStr, int loc)
      throws PatchFailedException {
    // If we're doing a renaming,
    // old file should be specified and exists,
    // new file should be specified but doesn't exist yet.
    String oldFileError = "";
    String newFileError = "";
    if (oldFile == null) {
      oldFileError = ", old file name (%s) is not specified";
    } else if (!oldFile.exists()) {
      oldFileError = String.format(", old file name (%s) doesn't exist", oldFileStr);
    }
    if (newFile == null) {
      newFileError = ", new file name is not specified";
    } else if (newFile.exists()) {
      newFileError = String.format(", new file name (%s) already exists", newFileStr);
    }
    if (!oldFileError.isEmpty() || !newFileError.isEmpty()) {
      throw new PatchFailedException(
          String.format("Cannot rename file (near line %d)%s%s.", loc, oldFileError, newFileError));
    }
  }

  private static void checkFilesStatusForPatching(
      Patch<String> patch,
      Path oldFile,
      Path newFile,
      String oldFileStr,
      String newFileStr,
      int loc)
      throws PatchFailedException {
    // At least one of oldFile or newFile should be specified.
    if (oldFile == null && newFile == null) {
      throw new PatchFailedException(
          String.format(
              "Wrong patch format near line %d, neither new file or old file are specified.", loc));
    }

    // Does this patch look like adding a new file.
    boolean isAddFile =
        patch.getDeltas().size() == 1 && patch.getDeltas().get(0).getSource().getLines().isEmpty();

    // If this patch is not adding a new file,
    // then either old file or new file should be specified and exists,
    // if not we throw an error.
    if (!isAddFile
        && (oldFile == null || !oldFile.exists())
        && (newFile == null || !newFile.exists())) {
      String oldFileError;
      String newFileError;
      if (oldFile == null) {
        oldFileError = ", old file name (%s) is not specified";
      } else {
        oldFileError = String.format(", old file name (%s) doesn't exist", oldFileStr);
      }
      if (newFile == null) {
        newFileError = ", new file name is not specified";
      } else {
        newFileError = String.format(", new file name (%s) doesn't exist", newFileStr);
      }
      throw new PatchFailedException(
          String.format(
              "Cannot find file to patch (near line %d)%s%s.", loc, oldFileError, newFileError));
    }
  }

  /**
   * Apply a patch file under a directory
   *
   * @param patchFile the patch file to apply
   * @param strip the number of leading components to strip from file path in the patch file
   * @param outputDirectory the repository directory to apply the patch file
   */
  public static void apply(Path patchFile, int strip, Path outputDirectory)
      throws IOException, PatchFailedException {
    if (!patchFile.exists()) {
      throw new PatchFailedException("Cannot find patch file: " + patchFile.getPathString());
    }

    boolean isGitDiff = false;
    boolean hasRenameFrom = false;
    boolean hasRenameTo = false;
    boolean isReadingChunk = false;
    List<String> patchContent = new ArrayList<>();
    ChunkHeader header = null;
    String oldFileStr = null;
    String newFileStr = null;
    Path oldFile = null;
    Path newFile = null;
    int oldLineCount = 0;
    int newLineCount = 0;
    int filePermission = -1;
    Result result;

    ImmutableList<String> patchFileLines = readFile(patchFile);
    for (int i = 0; i <= patchFileLines.size(); i++) {
      // Adding an extra line to make sure last chunk also gets applied.
      String line = i < patchFileLines.size() ? patchFileLines.get(i) : "$";
      LineType type;
      switch (type = getLineType(line, isReadingChunk, isGitDiff)) {
        case OLD_FILE:
          patchContent.add(line);
          oldFileStr = extractPath(line, strip, i + 1);
          oldFile = getFilePath(oldFileStr, outputDirectory, i + 1);
          break;
        case NEW_FILE:
          patchContent.add(line);
          newFileStr = extractPath(line, strip, i + 1);
          newFile = getFilePath(newFileStr, outputDirectory, i + 1);
          break;
        case NEW_MODE:
        case NEW_FILE_MODE:
          // The line should look like: "new mode 100755" or "new file mode 100755"
          // 7 is the file permission for owner, which is at index 12 or 17
          int index = type == LineType.NEW_MODE ? 12 : 17;
          char c = line.charAt(index);
          if (c < '0' || c > '7') {
            throw new PatchFailedException(
                "Wrong file mode format at line " + (i + 1) + ": " + line);
          }
          filePermission = Character.getNumericValue(c);
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
                "Wrong chunk detected near line "
                    + (i + 1)
                    + ": "
                    + line
                    + ", does not expect an added line here.");
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
                "Wrong chunk detected near line "
                    + (i + 1)
                    + ": "
                    + line
                    + ", does not expect a deleted line here.");
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
                "Wrong chunk detected near line "
                    + (i + 1)
                    + ": "
                    + line
                    + ", does not expect a context line here.");
          }
          break;
        case RENAME_FROM:
          hasRenameFrom = true;
          if (oldFileStr == null) {
            // len("rename from ") == 12
            oldFileStr = line.substring(12).trim();
            if (oldFileStr.isEmpty()) {
              throw new PatchFailedException(
                  String.format("Cannot determine file name from line %d:\n%s", i + 1, line));
            }
            oldFile = getFilePath(oldFileStr, outputDirectory, i + 1);
          }
          break;
        case RENAME_TO:
          hasRenameTo = true;
          if (newFileStr == null) {
            // len("rename to ") == 10
            newFileStr = line.substring(10).trim();
            if (newFileStr.isEmpty()) {
              throw new PatchFailedException(
                  String.format("Cannot determine file name from line %d:\n%s", i + 1, line));
            }
            newFile = getFilePath(newFileStr, outputDirectory, i + 1);
          }
          break;
        case OTHER_GIT_LINE:
          break;
        case GIT_HEADER:
        case UNKNOWN:
          // A git header line or an unknown line should trigger an action to apply collected
          // patch content to a file.

          // Renaming is a git only format
          boolean isRenaming = isGitDiff && hasRenameFrom && hasRenameTo;

          if (!patchContent.isEmpty() || isRenaming || filePermission != -1) {
            // We collected something useful, let's do some checks before applying the patch.
            int patchStartLocation = i + 1 - patchContent.size();

            checkPatchContentIsComplete(
                patchContent, header, oldLineCount, newLineCount, patchStartLocation);

            if (isRenaming) {
              checkFilesStatusForRenaming(
                  oldFile, newFile, oldFileStr, newFileStr, patchStartLocation);
            }

            Patch<String> patch = UnifiedDiffUtils.parseUnifiedDiff(patchContent);
            checkFilesStatusForPatching(
                patch, oldFile, newFile, oldFileStr, newFileStr, patchStartLocation);

            applyPatchToFile(patch, oldFile, newFile, isRenaming, filePermission);
          }

          patchContent.clear();
          header = null;
          oldFileStr = null;
          newFileStr = null;
          oldFile = null;
          newFile = null;
          filePermission = -1;
          oldLineCount = 0;
          newLineCount = 0;
          isReadingChunk = false;
          // If the new patch starts with "diff --git " then it's a git diff.
          isGitDiff = type == LineType.GIT_HEADER;
          if (isGitDiff) {
            // In case there is no line starting with +++ and --- (file permission change),
            // try to parse the file names from the line starting with "diff --git"
            List<String> args = Splitter.on(' ').splitToList(line);
            if (args.size() >= 4) {
              oldFileStr = stripPath(args.get(2), strip);
              if (!oldFileStr.isEmpty()) {
                oldFile = getFilePath(oldFileStr, outputDirectory, i + 1);
              }
              newFileStr = stripPath(args.get(3), strip);
              if (!newFileStr.isEmpty()) {
                newFile = getFilePath(newFileStr, outputDirectory, i + 1);
              }
            }
          }
          hasRenameFrom = false;
          hasRenameTo = false;
          break;
      }
    }
  }

  private PatchUtil() {}
}
