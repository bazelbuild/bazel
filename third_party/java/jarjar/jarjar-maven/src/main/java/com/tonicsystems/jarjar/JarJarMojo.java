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
package com.tonicsystems.jarjar;

import com.tonicsystems.jarjar.classpath.ClassPath;
import com.tonicsystems.jarjar.transform.jar.DefaultJarProcessor;
import com.tonicsystems.jarjar.transform.config.RulesFileParser;
import com.tonicsystems.jarjar.transform.JarTransformer;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;

public class JarJarMojo extends AbstractMojo {

    private File fromJar;
    private File toJar;
    private File rulesFile;
    private String rules;
    @Deprecated // Maven might need this for compatibility.
    private boolean verbose;

    @Override
    public void execute() throws MojoExecutionException {
        if (!((rulesFile == null || !rulesFile.exists()) ^ (rules == null)))
            throw new MojoExecutionException("Exactly one of rules or rulesFile is required");

        try {
            DefaultJarProcessor processor = new DefaultJarProcessor();
            if (rules != null) {
                RulesFileParser.parse(processor, rules);
            } else {
                RulesFileParser.parse(processor, rulesFile);
            }
            // TODO: refactor with Main.java
            JarTransformer transformer = new JarTransformer(toJar, processor);
            ClassPath fromClassPath = new ClassPath(new File(System.getProperty("user.dir")), Collections.singleton(fromJar));
            transformer.transform(fromClassPath);
        } catch (IOException e) {
            throw new MojoExecutionException(e.getMessage(), e);
        }
    }
}
