package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;

public interface MetadataCache {
  void putFileMetadata(Artifact artifact, FileArtifactValue metadata);

  void removeFileMetadata(Artifact artifact);

  FileArtifactValue getFileMetadata(Artifact artifact);
}
