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
package proguard.io;

import java.io.*;

/**
 * This interface describes a data entry, e.g. a ZIP entry, a file, or a
 * directory.
 *
 * @author Eric Lafortune
 */
public interface DataEntry
{
    /**
     * Returns the name of this data entry.
     */
    public String getName();


    /**
     * Returns the original name of this data entry, i.e. the name of the
     * data entry before any renaming or obfuscation.
     */
    public String getOriginalName();


    /**
     * Returns the size of this data entry, in bytes, or -1 if unknown.
     */
    public long getSize();


    /**
     * Returns whether the data entry represents a directory.
     */
    public boolean isDirectory();


    /**
     * Returns an input stream for reading the content of this data entry.
     * The data entry may not represent a directory.
     */
    public InputStream getInputStream() throws IOException;


    /**
     * Closes the previously retrieved InputStream.
     */
    public void closeInputStream() throws IOException;


    /**
     * Returns the parent of this data entry, or <code>null</null> if it doesn't
     * have one.
     */
    public DataEntry getParent();
}
