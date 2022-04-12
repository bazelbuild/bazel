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
 * This DataEntryWriter delegates to a given DataEntryWriter, each time
 * adding a prefix of the written data entry name.
 *
 * @author Eric Lafortune
 */
public class PrefixAddingDataEntryWriter implements DataEntryWriter
{
    private final String          prefix;
    private final DataEntryWriter dataEntryWriter;


    /**
     * Creates a new PrefixAddingDataEntryWriter.
     */
    public PrefixAddingDataEntryWriter(String          prefix,
                                       DataEntryWriter dataEntryWriter)
    {
        this.prefix          = prefix;
        this.dataEntryWriter = dataEntryWriter;
    }


    // Implementations for DataEntryWriter.

    public boolean createDirectory(DataEntry dataEntry)
    throws IOException
    {
        return dataEntryWriter.createDirectory(renamedDataEntry(dataEntry));
    }


    public boolean sameOutputStream(DataEntry dataEntry1,
                                    DataEntry dataEntry2)
    throws IOException
    {
        return dataEntryWriter.sameOutputStream(renamedDataEntry(dataEntry1),
                                                renamedDataEntry(dataEntry2));
    }


    public OutputStream createOutputStream(DataEntry dataEntry)
    throws IOException
    {
        return dataEntryWriter.createOutputStream(renamedDataEntry(dataEntry));
    }


    public void close() throws IOException
    {
        dataEntryWriter.close();
    }


    public void println(PrintWriter pw, String prefix)
    {
        pw.println(prefix + "PrefixAddingDataEntryWriter (prefix = "+prefix+")");
        dataEntryWriter.println(pw, prefix + "  ");
    }


    // Small utility methods.

    /**
     * Adds the prefix to the given data entry name.
     */
    private DataEntry renamedDataEntry(DataEntry dataEntry)
    {
        return new RenamedDataEntry(dataEntry, prefix + dataEntry.getName());
    }
}
