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

import com.tonicsystems.jarjar.transform.asm.PackageRemapper;
import com.tonicsystems.jarjar.transform.Transformable;
import com.tonicsystems.jarjar.util.ClassNameUtils;
import java.io.IOException;
import javax.annotation.Nonnull;

/**
 * Allows any file which is NOT a JAR file.
 */
public class ResourceRenamerJarProcessor implements JarProcessor {

    private final PackageRemapper pr;

    public ResourceRenamerJarProcessor(@Nonnull PackageRemapper pr) {
        this.pr = pr;
    }

    @Override
    public Result scan(Transformable struct) throws IOException {
        return Result.KEEP;
    }

    @Override
    public Result process(Transformable struct) throws IOException {
        if (!ClassNameUtils.isClass(struct.name))
            struct.name = pr.mapPath(struct.name);
        return Result.KEEP;
    }
}
