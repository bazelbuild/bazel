package com.google.devtools.build.lib.bazel.execlog;

import com.google.devtools.build.lib.exec.Protos;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.zip.GZIPOutputStream;
import java.util.zip.GZIPInputStream;

/**
 * Stores execution log history in the file system for incremental execution log analysis
 */
class ExeclogHistory {

  private final Path dir;

  ExeclogHistory(Path outputBase) {
    dir = outputBase.getChild("execlog_history");
  }

  void put(Protos.SpawnExec spawn) throws IOException {
    Path path = getPath(spawn);
    path.getParentDirectory().createDirectoryAndParents();
    try (GZIPOutputStream out = new GZIPOutputStream(path.getOutputStream())) {
      spawn.writeTo(out);
    }
  }

  Protos.SpawnExec get(Protos.SpawnExec spawn) throws IOException {
    Path path = getPath(spawn);
    if (!path.exists()) {
      return null;
    }
    try (GZIPInputStream in = new GZIPInputStream(path.getInputStream())) {
      return Protos.SpawnExec.parseFrom(in);
    }
  }

  private Path getPath(Protos.SpawnExec spawn) {
    String key = spawn.getTargetLabel() + '|' + spawn.getProgressMessage(); // should be unique
    String digest = Fingerprint.getHexDigest(key);
    return dir.getChild(digest.substring(0, 2)).getChild(digest);
  }
}
