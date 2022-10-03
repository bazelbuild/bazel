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
package com.google.devtools.build.lib.actions.usage;

import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.vfs.Path;

/**
 * Implementation of FileArtifactValue using pre-computed hardcoded value.
 */
public final class PreComputedMetadataValue extends FileArtifactValue {
    private byte[] digest;

    PreComputedMetadataValue(byte[] digest) {
        this.digest = digest;
    }

    @Override
    public FileStateType getType() {
        return FileStateType.REGULAR_FILE;
    }

    @Override
    public byte[] getDigest() {
        return digest;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
        throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
        return 0;
    }

    @Override
    public long getModifiedTime() {
        return -1;
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
        throw new UnsupportedOperationException(
                "PreComputedMetadataValue doesn't support wasModifiedSinceDigest " + path.toString());
    }
}
