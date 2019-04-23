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
 * A <code>WordReader</code> that returns words from a line number reader.
 *
 * @author Eric Lafortune
 */
public class LineWordReader extends WordReader
{
    private final LineNumberReader reader;
    private final String           description;


    /**
     * Creates a new LineWordReader for the given input.
     */
    public LineWordReader(LineNumberReader lineNumberReader,
                          String           description,
                          File             baseDir) throws IOException
    {
        super(baseDir);

        this.reader      = lineNumberReader;
        this.description = description;
    }


    /**
     * Creates a new LineWordReader for the given input.
     */
    public LineWordReader(LineNumberReader lineNumberReader,
                          String           description,
                          URL              baseURL) throws IOException
    {
        super(baseURL);

        this.reader      = lineNumberReader;
        this.description = description;
    }


    // Implementations for WordReader.

    protected String nextLine() throws IOException
    {
        return reader.readLine();
    }


    protected String lineLocationDescription()
    {
        return "line " + reader.getLineNumber() + " of " + description;
    }


    public void close() throws IOException
    {
        super.close();

        if (reader != null)
        {
            reader.close();
        }
    }
}
