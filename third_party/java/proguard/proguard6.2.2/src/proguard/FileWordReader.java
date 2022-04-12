/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard;

import java.io.*;
import java.net.URL;


/**
 * A <code>WordReader</code> that returns words from a file or a URL.
 *
 * @author Eric Lafortune
 */
public class FileWordReader extends LineWordReader
{
    /**
     * Creates a new FileWordReader for the given file.
     */
    public FileWordReader(File file) throws IOException
    {
        super(new LineNumberReader(
              new BufferedReader(
              new InputStreamReader(
              new FileInputStream(file), "UTF-8"))),
              "file '" + file.getPath() + "'",
              file.getParentFile());
    }


    /**
     * Creates a new FileWordReader for the given URL.
     */
    public FileWordReader(URL url) throws IOException
    {
        super(new LineNumberReader(
              new BufferedReader(
              new InputStreamReader(url.openStream(), "UTF-8"))),
              "file '" + url.toString() + "'",
              url);
    }
}
