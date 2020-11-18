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
 * This DataEntryWriter delegates to one of two other DataEntryWriter instances,
 * depending on whether the data entry passes through a given data entry filter
 * or not.
 *
 * @author Eric Lafortune
 */
public class FilteredDataEntryWriter implements DataEntryWriter
{
    private final DataEntryFilter dataEntryFilter;
    private DataEntryWriter acceptedDataEntryWriter;
    private DataEntryWriter rejectedDataEntryWriter;


    /**
     * Creates a new FilteredDataEntryWriter with only a writer for accepted
     * data entries.
     * @param dataEntryFilter         the data entry filter.
     * @param acceptedDataEntryWriter the DataEntryWriter to which the writing
     *                                will be delegated if the filter accepts
     *                                the data entry. May be <code>null</code>.
     */
    public FilteredDataEntryWriter(DataEntryFilter dataEntryFilter,
                                   DataEntryWriter acceptedDataEntryWriter)
    {
        this(dataEntryFilter, acceptedDataEntryWriter, null);
    }


    /**
     * Creates a new FilteredDataEntryWriter.
     * @param dataEntryFilter         the data entry filter.
     * @param acceptedDataEntryWriter the DataEntryWriter to which the writing
     *                                will be delegated if the filter accepts
     *                                the data entry. May be <code>null</code>.
     * @param rejectedDataEntryWriter the DataEntryWriter to which the writing
     *                                will be delegated if the filter does not
     *                                accept the data entry. May be
     *                                <code>null</code>.
     */
    public FilteredDataEntryWriter(DataEntryFilter dataEntryFilter,
                                   DataEntryWriter acceptedDataEntryWriter,
                                   DataEntryWriter rejectedDataEntryWriter)
    {
        this.dataEntryFilter         = dataEntryFilter;
        this.acceptedDataEntryWriter = acceptedDataEntryWriter;
        this.rejectedDataEntryWriter = rejectedDataEntryWriter;
    }


    // Implementations for DataEntryWriter.

    public boolean createDirectory(DataEntry dataEntry) throws IOException
    {
        // Get the right data entry writer.
        DataEntryWriter dataEntryWriter = dataEntryFilter.accepts(dataEntry) ?
            acceptedDataEntryWriter :
            rejectedDataEntryWriter;

        // Delegate to it, if it's not null.
        return dataEntryWriter != null &&
               dataEntryWriter.createDirectory(dataEntry);
    }


    public boolean sameOutputStream(DataEntry dataEntry1,
                                    DataEntry dataEntry2)
    throws IOException
    {
        boolean accepts1 = dataEntryFilter.accepts(dataEntry1);
        boolean accepts2 = dataEntryFilter.accepts(dataEntry2);
        return
            accepts1 ? !accepts2 || acceptedDataEntryWriter == null || acceptedDataEntryWriter.sameOutputStream(dataEntry1, dataEntry2) :
                       accepts2  || rejectedDataEntryWriter == null || rejectedDataEntryWriter.sameOutputStream(dataEntry1, dataEntry2);
    }


    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException
    {
        // Get the right data entry writer.
        DataEntryWriter dataEntryWriter = dataEntryFilter.accepts(dataEntry) ?
            acceptedDataEntryWriter :
            rejectedDataEntryWriter;

        // Delegate to it, if it's not null.
        return dataEntryWriter != null ?
            dataEntryWriter.createOutputStream(dataEntry) :
            null;
    }


    public void close() throws IOException
    {
        if (acceptedDataEntryWriter != null)
        {
            acceptedDataEntryWriter.close();
            acceptedDataEntryWriter = null;
        }

        if (rejectedDataEntryWriter != null)
        {
            rejectedDataEntryWriter.close();
            rejectedDataEntryWriter = null;
        }
    }


    public void println(PrintWriter pw, String prefix)
    {
        pw.println(prefix + "FilteredDataEntryWriter (filter = "+dataEntryFilter+")");
        if (acceptedDataEntryWriter != null)
        {
            acceptedDataEntryWriter.println(pw, prefix + "  ");
        }
        if (rejectedDataEntryWriter != null)
        {
            rejectedDataEntryWriter.println(pw, prefix + "  ");
        }
    }
}
