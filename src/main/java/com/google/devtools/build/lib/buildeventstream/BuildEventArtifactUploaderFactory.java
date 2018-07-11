// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildeventstream;

import static com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.LOCAL_FILES_UPLOADER;

import com.google.devtools.common.options.OptionsProvider;

/** A factory for {@link BuildEventArtifactUploader}. */
public interface BuildEventArtifactUploaderFactory {

  BuildEventArtifactUploaderFactory LOCAL_FILES_UPLOADER_FACTORY =
      (OptionsProvider options) -> LOCAL_FILES_UPLOADER;

  /**
   * Returns a new instance of a {@link BuildEventArtifactUploader}. The call is responsible for
   * calling {@link BuildEventArtifactUploader#shutdown()} on the returned instance.
   */
  BuildEventArtifactUploader create(OptionsProvider options);
}
