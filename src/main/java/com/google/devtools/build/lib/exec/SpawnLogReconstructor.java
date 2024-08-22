// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.github.luben.zstd.ZstdInputStream;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.exec.Protos.ExecLogEntry;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.annotation.Nullable;

/** Reconstructs an execution log in expanded format from the compact format representation. */
public final class SpawnLogReconstructor implements MessageInputStream<SpawnExec> {
  private final ZstdInputStream in;

  private final HashMap<Integer, File> fileMap = new HashMap<>();
  private final HashMap<Integer, Pair<String, Collection<File>>> dirMap = new HashMap<>();
  private final HashMap<Integer, File> symlinkMap = new HashMap<>();
  private final HashMap<Integer, ExecLogEntry.InputSet> setMap = new HashMap<>();
  private String hashFunctionName = "";

  public SpawnLogReconstructor(InputStream in) throws IOException {
    this.in = new ZstdInputStream(in);
  }

  @Override
  @Nullable
  public SpawnExec read() throws IOException {
    ExecLogEntry entry;
    while ((entry = ExecLogEntry.parseDelimitedFrom(in)) != null) {
      switch (entry.getTypeCase()) {
        case INVOCATION:
          hashFunctionName = entry.getInvocation().getHashFunctionName();
          break;
        case FILE:
          fileMap.put(entry.getId(), reconstructFile(entry.getFile()));
          break;
        case DIRECTORY:
          dirMap.put(entry.getId(), reconstructDir(entry.getDirectory()));
          break;
        case UNRESOLVED_SYMLINK:
          symlinkMap.put(entry.getId(), reconstructSymlink(entry.getUnresolvedSymlink()));
          break;
        case INPUT_SET:
          setMap.put(entry.getId(), entry.getInputSet());
          break;
        case SPAWN:
          return reconstructSpawnExec(entry.getSpawn());
        default:
          throw new IOException(
              String.format("unknown entry type %d", entry.getTypeCase().getNumber()));
        case SYMLINK_ACTION -> {
          // Symlink actions are not represented in the expanded format.
        }
      }
    }
    return null;
  }

  private SpawnExec reconstructSpawnExec(ExecLogEntry.Spawn entry) throws IOException {
    SpawnExec.Builder builder =
        SpawnExec.newBuilder()
            .addAllCommandArgs(entry.getArgsList())
            .addAllEnvironmentVariables(entry.getEnvVarsList())
            .setTargetLabel(entry.getTargetLabel())
            .setMnemonic(entry.getMnemonic())
            .setExitCode(entry.getExitCode())
            .setStatus(entry.getStatus())
            .setRunner(entry.getRunner())
            .setCacheHit(entry.getCacheHit())
            .setRemotable(entry.getRemotable())
            .setCacheable(entry.getCacheable())
            .setRemoteCacheable(entry.getRemoteCacheable())
            .setTimeoutMillis(entry.getTimeoutMillis())
            .setMetrics(entry.getMetrics());

    if (entry.hasPlatform()) {
      builder.setPlatform(entry.getPlatform());
    }

    SortedMap<String, File> inputs = reconstructInputs(entry.getInputSetId());
    SortedMap<String, File> toolInputs = reconstructInputs(entry.getToolSetId());

    for (Map.Entry<String, File> e : inputs.entrySet()) {
      File file = e.getValue();
      if (toolInputs.containsKey(e.getKey())) {
        file = file.toBuilder().setIsTool(true).build();
      }
      builder.addInputs(file);
    }

    SortedSet<String> listedOutputs = new TreeSet<>();

    for (ExecLogEntry.Output output : entry.getOutputsList()) {
      switch (output.getTypeCase()) {
        case FILE_ID:
          File file = getFromMap(fileMap, output.getFileId());
          listedOutputs.add(file.getPath());
          builder.addActualOutputs(file);
          break;
        case DIRECTORY_ID:
          Pair<String, Collection<File>> dir = getFromMap(dirMap, output.getDirectoryId());
          listedOutputs.add(dir.getFirst());
          for (File dirFile : dir.getSecond()) {
            builder.addActualOutputs(dirFile);
          }
          break;
        case UNRESOLVED_SYMLINK_ID:
          File symlink = getFromMap(symlinkMap, output.getUnresolvedSymlinkId());
          listedOutputs.add(symlink.getPath());
          builder.addActualOutputs(symlink);
          break;
        case INVALID_OUTPUT_PATH:
          listedOutputs.add(output.getInvalidOutputPath());
          break;
        default:
          throw new IOException(
              String.format("unknown output type %d", output.getTypeCase().getNumber()));
      }
    }

    builder.addAllListedOutputs(listedOutputs);

    if (entry.hasDigest()) {
      builder.setDigest(entry.getDigest().toBuilder().setHashFunctionName(hashFunctionName));
    }

    return builder.build();
  }

  private SortedMap<String, File> reconstructInputs(int setId) throws IOException {
    TreeMap<String, File> inputs = new TreeMap<>();
    ArrayDeque<Integer> setsToVisit = new ArrayDeque<>();
    HashSet<Integer> visited = new HashSet<>();
    if (setId != 0) {
      setsToVisit.addLast(setId);
      visited.add(setId);
    }
    while (!setsToVisit.isEmpty()) {
      ExecLogEntry.InputSet set = getFromMap(setMap, setsToVisit.removeFirst());
      for (int fileId : set.getFileIdsList()) {
        if (visited.add(fileId)) {
          File file = getFromMap(fileMap, fileId);
          inputs.put(file.getPath(), file);
        }
      }
      for (int dirId : set.getDirectoryIdsList()) {
        if (visited.add(dirId)) {
          Pair<String, Collection<File>> dir = getFromMap(dirMap, dirId);
          for (File dirFile : dir.getSecond()) {
            inputs.put(dirFile.getPath(), dirFile);
          }
        }
      }
      for (int symlinkId : set.getUnresolvedSymlinkIdsList()) {
        if (visited.add(symlinkId)) {
          File symlink = getFromMap(symlinkMap, symlinkId);
          inputs.put(symlink.getPath(), symlink);
        }
      }
      for (int transitiveSetId : set.getTransitiveSetIdsList()) {
        if (visited.add(transitiveSetId)) {
          setsToVisit.addLast(transitiveSetId);
        }
      }
    }
    return inputs;
  }

  private Pair<String, Collection<File>> reconstructDir(ExecLogEntry.Directory dir) {
    ImmutableList.Builder<File> builder =
        ImmutableList.builderWithExpectedSize(dir.getFilesCount());
    for (ExecLogEntry.File dirFile : dir.getFilesList()) {
      builder.add(reconstructFile(dir, dirFile));
    }
    return Pair.of(dir.getPath(), builder.build());
  }

  private File reconstructFile(ExecLogEntry.File entry) {
    return reconstructFile(null, entry);
  }

  private File reconstructFile(
      @Nullable ExecLogEntry.Directory parentDir, ExecLogEntry.File entry) {
    File.Builder builder = File.newBuilder();
    builder.setPath(
        parentDir != null ? parentDir.getPath() + "/" + entry.getPath() : entry.getPath());
    if (entry.hasDigest()) {
      builder.setDigest(entry.getDigest().toBuilder().setHashFunctionName(hashFunctionName));
    }
    return builder.build();
  }

  private static File reconstructSymlink(ExecLogEntry.UnresolvedSymlink entry) {
    return File.newBuilder()
        .setPath(entry.getPath())
        .setSymlinkTargetPath(entry.getTargetPath())
        .build();
  }

  private static <T> T getFromMap(Map<Integer, T> map, int id) throws IOException {
    T value = map.get(id);
    if (value == null) {
      throw new IOException(String.format("referenced entry %d is missing or has wrong type", id));
    }
    return value;
  }

  @Override
  public void close() throws IOException {
    in.close();
  }
}
