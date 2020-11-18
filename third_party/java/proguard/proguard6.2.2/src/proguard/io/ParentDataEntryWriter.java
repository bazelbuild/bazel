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
 * This DataEntryWriter lets another DataEntryWriter write the parent data
 * entries.
 *
 * @author Eric Lafortune
 */
public class ParentDataEntryWriter implements DataEntryWriter
{
    private DataEntryWriter dataEntryWriter;


    /**
     * Creates a new ParentDataEntryWriter.
     * @param dataEntryWriter the DataEntryWriter to which the writing will be
     *                        delegated, passing the data entries' parents.
     */
    public ParentDataEntryWriter(DataEntryWriter dataEntryWriter)
    {
        this.dataEntryWriter = dataEntryWriter;
    }


    // Implementations for DataEntryWriter.

    public boolean createDirectory(DataEntry dataEntry) throws IOException
    {
        return dataEntryWriter.createDirectory(dataEntry.getParent());
    }


    public boolean sameOutputStream(DataEntry dataEntry1,
                                    DataEntry dataEntry2)
    throws IOException
    {
        return dataEntryWriter.sameOutputStream(dataEntry1.getParent(),
                                                dataEntry2.getParent());
    }


    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException
    {
        return dataEntryWriter.createOutputStream(dataEntry.getParent());
    }


    public void close() throws IOException
    {
        dataEntryWriter.close();
        dataEntryWriter = null;
    }


    public void println(PrintWriter pw, String prefix)
    {
        pw.println(prefix + "ParentDataEntryWriter");
        dataEntryWriter.println(pw, prefix + "  ");
    }
}
