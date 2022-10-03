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
import com.google.devtools.build.lib.view.proto.Deps;

import javax.annotation.Nullable;

/**
 * Class holding info for an input artifact internal class being used by an action.
 */
public class ClassUsageInfo {
    private String fullyQualifiedName;
    private String internalPath;
    private FileArtifactValue compileTimeFileArtifactValue;
    @Nullable
    private FileArtifactValue dependencyFileArtifactValue;

    private ClassUsageInfo() {
    }

    static ClassUsageInfo create(Deps.UsedClass c) {
        return create(c.getFullyQualifiedName(),
                c.getJarInternalPath(),
                new PreComputedMetadataValue(c.getHash().toByteArray()),
                /* dependencyFileArtifactValue= */ null);
    }

    static ClassUsageInfo create(String fullyQualifiedName, String internalPath, FileArtifactValue compileTimeFileArtifactValue, FileArtifactValue dependencyFileArtifactValue) {
        ClassUsageInfo usageInfo = new ClassUsageInfo();
        usageInfo.fullyQualifiedName = fullyQualifiedName;
        usageInfo.internalPath = internalPath;
        usageInfo.compileTimeFileArtifactValue = compileTimeFileArtifactValue;
        usageInfo.dependencyFileArtifactValue = dependencyFileArtifactValue;
        return usageInfo;
    }

    /**
     * Fully qualified name of class being used.
     */
    public String getFullyQualifiedName() {
        return fullyQualifiedName;
    }

    /**
     * Path of .class file within compiled (abi) jar.
     */
    public String getInternalPath() {
        return internalPath;
    }

    /**
     * Hash of .class file as to when it was used during compilation.
     */
    public FileArtifactValue getCompileTimeFileArtifactValue() {
        return compileTimeFileArtifactValue;
    }

    /**
     * Hash of .class file as extracted by input artifact jar prior to compilation.
     */
    public FileArtifactValue getDependencyFileArtifactValue() {
        return dependencyFileArtifactValue;
    }
}
