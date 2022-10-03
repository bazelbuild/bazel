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

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.DigestHashFunction;

import javax.annotation.Nullable;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;

/**
 * Utility class for action input tracking.
 */
class Utils {

    /**
     * Compute sha256 of the jar entry corresponding to provided path.
     */
    static byte[] getHashFromJarEntry(Artifact artifact, String path) {
        try (JarFile jarFile = new JarFile(artifact.getPath().getPathFile())) {
            ZipEntry entry = jarFile.getEntry(path);
            if (entry == null) {
                return null;
            }
            InputStream stream = jarFile.getInputStream(entry);
            byte[] targetArray = ByteStreams.toByteArray(stream);
            return DigestHashFunction.SHA256.getHashFunction().hashBytes(targetArray).asBytes();
        } catch (IOException e) {
        }
        return new byte[0];
    }

    /**
     * Get action .jdeps artifact, used to extract compilation tracking information from.
     */
    @Nullable
    static Artifact getJDepsOutput(Action action) {
        List<Artifact> jdepsOutput = action.getOutputs().stream()
                .filter(output -> output.getExecPathString().endsWith(".jdeps"))
                .collect(Collectors.toList());
        return jdepsOutput.size() == 1 ? jdepsOutput.get(0) : null;
    }
}
