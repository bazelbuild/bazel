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
 * This DataEntry wraps another data entry.
 *
 * @author Thomas Neidhart
 */
public class WrappedDataEntry implements DataEntry
{
    protected final DataEntry wrappedEntry;


    public WrappedDataEntry(DataEntry wrappedEntry)
    {
        this.wrappedEntry = wrappedEntry;
    }


    public void closeInputStream() throws IOException
    {
        wrappedEntry.closeInputStream();
    }

    public String getName()
    {
        return wrappedEntry.getName();
    }


    public String getOriginalName()
    {
        return wrappedEntry.getOriginalName();
    }


    public long getSize()
    {
        return wrappedEntry.getSize();
    }


    public boolean isDirectory()
    {
        return wrappedEntry.isDirectory();
    }


    public InputStream getInputStream() throws IOException
    {
        return wrappedEntry.getInputStream();
    }


    public DataEntry getParent()
    {
        return wrappedEntry.getParent();
    }


    public String toString()
    {
        return getName();
    }

}
