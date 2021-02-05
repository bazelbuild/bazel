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
 * This DataEntryWriter delegates to a given DataEntryWriter, or failing that,
 * to another given DataEntryWriter.
 *
 * @author Eric Lafortune
 */
public class CascadingDataEntryWriter implements DataEntryWriter
{
    private DataEntryWriter dataEntryWriter1;
    private DataEntryWriter dataEntryWriter2;


    /**
     * Creates a new CascadingDataEntryWriter.
     * @param dataEntryWriter1 the DataEntryWriter to which the writing will be
     *                         delegated first.
     * @param dataEntryWriter2 the DataEntryWriter to which the writing will be
     *                         delegated, if the first one can't provide an
     *                         output stream.
     */
    public CascadingDataEntryWriter(DataEntryWriter dataEntryWriter1,
                                    DataEntryWriter dataEntryWriter2)
    {
        this.dataEntryWriter1 = dataEntryWriter1;
        this.dataEntryWriter2 = dataEntryWriter2;
    }


    // Implementations for DataEntryWriter.

    public boolean createDirectory(DataEntry dataEntry) throws IOException
    {
        // Try to create a directory with the first data entry writer, or
        // otherwise with the second data entry writer.
        return dataEntryWriter1.createDirectory(dataEntry) ||
               dataEntryWriter2.createDirectory(dataEntry);
    }


    public boolean sameOutputStream(DataEntry dataEntry1,
                                    DataEntry dataEntry2)
    throws IOException
    {
        return dataEntryWriter1.sameOutputStream(dataEntry1, dataEntry2) ||
               dataEntryWriter2.sameOutputStream(dataEntry1, dataEntry2);
    }


    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException
    {
        // Try to get an output stream from the first data entry writer.
        OutputStream outputStream =
            dataEntryWriter1.createOutputStream(dataEntry);

        // Return it, if it's not null. Otherwise try to get an output stream
        // from the second data entry writer.
        return outputStream != null ?
            outputStream :
            dataEntryWriter2.createOutputStream(dataEntry);
    }


    public void close() throws IOException
    {
        dataEntryWriter1.close();
        dataEntryWriter2.close();

        dataEntryWriter1 = null;
        dataEntryWriter2 = null;
    }


    public void println(PrintWriter pw, String prefix)
    {
        pw.println(prefix + "CascadingDataEntryWriter");
        dataEntryWriter1.println(pw, prefix + "  ");
        dataEntryWriter2.println(pw, prefix + "  ");
    }
}
