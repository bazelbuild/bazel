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
package com.tonicsystems.jarjar.util;

import com.tonicsystems.jarjar.transform.Transformable;
import com.tonicsystems.jarjar.transform.jar.JarProcessor;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.Set;
import org.apache.tools.ant.BuildException;
import org.apache.tools.ant.taskdefs.Jar;
import org.apache.tools.ant.types.ZipFileSet;
import org.apache.tools.zip.JarMarker;
import org.apache.tools.zip.ZipExtraField;
import org.apache.tools.zip.ZipOutputStream;

public abstract class AntJarProcessor extends Jar {

    private final Transformable struct = new Transformable();
    private JarProcessor proc;
    private final byte[] buf = new byte[0x2000];

    private final Set<String> dirs = new HashSet<String>();
    private boolean filesOnly;

    protected boolean verbose;

    private static final ZipExtraField[] JAR_MARKER = new ZipExtraField[]{
        JarMarker.getInstance()
    };

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    @Override
    public abstract void execute() throws BuildException;

    public void execute(JarProcessor proc) throws BuildException {
        this.proc = proc;
        super.execute();
    }

    @Override
    public void setFilesonly(boolean f) {
        super.setFilesonly(f);
        filesOnly = f;
    }

    @Override
    protected void zipDir(File dir, ZipOutputStream zOut, String vPath, int mode)
            throws IOException {
    }

    // TODO: Rewrite this entirely.
    @Override
    protected void zipFile(InputStream is, ZipOutputStream zOut, String vPath,
            long lastModified, File fromArchive, int mode) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        IoUtil.copy(is, baos, buf);
        struct.data = baos.toByteArray();
        struct.name = vPath;
        struct.time = lastModified;
        if (proc.process(struct) != JarProcessor.Result.DISCARD) {
            if (mode == 0)
                mode = ZipFileSet.DEFAULT_FILE_MODE;
            if (!filesOnly) {
                addParentDirs(struct.name, zOut);
            }
            super.zipFile(new ByteArrayInputStream(struct.data),
                    zOut, struct.name, struct.time, fromArchive, mode);
        }
    }

    private void addParentDirs(String file, ZipOutputStream zOut) throws IOException {
        int slash = file.lastIndexOf('/');
        if (slash >= 0) {
            String dir = file.substring(0, slash);
            if (dirs.add(dir)) {
                addParentDirs(dir, zOut);
                super.zipDir((File) null, zOut, dir + "/", ZipFileSet.DEFAULT_DIR_MODE, JAR_MARKER);
            }
        }
    }

    @Override
    public void reset() {
        super.reset();
        cleanHelper();
    }

    @Override
    protected void cleanUp() {
        super.cleanUp();
        cleanHelper();
    }

    protected void cleanHelper() {
        verbose = false;
        filesOnly = false;
        dirs.clear();
    }
}
