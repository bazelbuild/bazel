/**
 * Copyright 2007 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.tonicsystems.jarjar.transform.jar;

import java.util.Collections;

/**
 * Excludes the manifest.
 */
public class ManifestFilterJarProcessor extends PathFilterJarProcessor {

    public static final String MANIFEST_PATH = "META-INF/MANIFEST.MF";

    private boolean enabled = false;

    public ManifestFilterJarProcessor() {
        super(Collections.singleton(MANIFEST_PATH));
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    @Override
    protected boolean isFiltered(String name) {
        if (!isEnabled())
            return false;
        return super.isFiltered(name);
    }

}
