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

import com.tonicsystems.jarjar.transform.config.ClassDelete;
import com.tonicsystems.jarjar.transform.config.ClassKeepTransitive;
import com.tonicsystems.jarjar.transform.config.ClassRename;
import com.tonicsystems.jarjar.transform.jar.DefaultJarProcessor;
import com.tonicsystems.jarjar.util.AntJarProcessor;
import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;
import org.apache.tools.ant.BuildException;

public class JarJarTask extends AntJarProcessor {

    private DefaultJarProcessor processor = new DefaultJarProcessor();

    @Nonnull
    private static String checkNotNull(@CheckForNull String in, @Nonnull String msg) {
        if (in == null)
            throw new IllegalArgumentException(msg);
        return in;
    }

    public void addConfiguredRule(Rule rule) {
        processor.addClassRename(new ClassRename(
                checkNotNull(rule.getPattern(), "The <rule> element requires the \"pattern\" attribute."),
                checkNotNull(rule.getResult(), "The <rule> element requires the \"result\" attribute.")
        ));
    }

    public void addConfiguredZap(Zap zap) {
        processor.addClassDelete(new ClassDelete(
                checkNotNull(zap.getPattern(), "The <zap> element requires a \"pattern\" attribute.")
        ));
    }

    public void addConfiguredKeep(Keep keep) {
        processor.addClassKeepTransitive(new ClassKeepTransitive(
                checkNotNull(keep.getPattern(), "The <keep> element requires a \"pattern\" attribute.")
        ));
    }

    @Override
    public void execute() throws BuildException {
        execute(processor);
    }

    @Override
    protected void cleanHelper() {
        super.cleanHelper();
        processor = new DefaultJarProcessor();
    }
}
