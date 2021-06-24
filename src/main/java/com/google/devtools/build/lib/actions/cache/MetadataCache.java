package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;

public interface MetadataCache {
  void putFileMetadata(Artifact artifact, FileArtifactValue metadata);

  void removeFileMetadata(Artifact artifact);

  FileArtifactValue getFileMetadata(Artifact artifact);

  void putTreeMetadata(SpecialArtifact artifact, TreeArtifactValue metadata);

  void removeTreeMetadata(SpecialArtifact artifact);

  TreeArtifactValue getTreeMetadata(SpecialArtifact artifact);
}
