package com.google.devtools.build.lib.buildeventstream;

import static com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.LOCAL_FILES_UPLOADER;

public interface BuildEventArtifactUploaderFactory {

  BuildEventArtifactUploaderFactory LOCAL_FILES_UPLOADER_FACTORY = () -> LOCAL_FILES_UPLOADER;

  BuildEventArtifactUploader create();

}
