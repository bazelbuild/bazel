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

import javax.annotation.Nullable;
import java.util.Set;

/**
 * Class holding tracking info for a given action/input artifact pair.
 */
public class TrackingInfo {

    private boolean isUnused;

    @Nullable
    private Set<ClassUsageInfo> usedClasses;

    public TrackingInfo(boolean isUnused, Set<ClassUsageInfo> usedClasses) {
        this.isUnused = isUnused;
        this.usedClasses = usedClasses;
    }

    public boolean isUnused() {
        return isUnused;
    }

    public boolean tracksUsedClasses() {
        return usedClasses != null;
    }

    public Set<ClassUsageInfo> getUsedClasses() {
        return usedClasses;
    }
}
