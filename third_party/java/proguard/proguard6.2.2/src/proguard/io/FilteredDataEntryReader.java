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

import java.io.IOException;


/**
 * This DataEntryReader delegates to one of two other DataEntryReader instances,
 * depending on whether the data entry passes through a given data entry filter
 * or not.
 *
 * @author Eric Lafortune
 */
public class FilteredDataEntryReader implements DataEntryReader
{
    private final DataEntryFilter dataEntryFilter;
    private final DataEntryReader acceptedDataEntryReader;
    private final DataEntryReader rejectedDataEntryReader;


    /**
     * Creates a new FilteredDataEntryReader with only a reader for accepted
     * data entries.
     * @param dataEntryFilter         the data entry filter.
     * @param acceptedDataEntryReader the DataEntryReader to which the reading
     *                                will be delegated if the filter accepts
     *                                the data entry. May be <code>null</code>.
     */
    public FilteredDataEntryReader(DataEntryFilter dataEntryFilter,
                                   DataEntryReader acceptedDataEntryReader)
    {
        this(dataEntryFilter, acceptedDataEntryReader, null);
    }


    /**
     * Creates a new FilteredDataEntryReader.
     * @param dataEntryFilter         the data entry filter.
     * @param acceptedDataEntryReader the DataEntryReader to which the reading
     *                                will be delegated if the filter accepts
     *                                the data entry. May be <code>null</code>.
     * @param rejectedDataEntryReader the DataEntryReader to which the reading
     *                                will be delegated if the filter does not
     *                                accept the data entry. May be
     *                                <code>null</code>.
     */
    public FilteredDataEntryReader(DataEntryFilter dataEntryFilter,
                                   DataEntryReader acceptedDataEntryReader,
                                   DataEntryReader rejectedDataEntryReader)
    {
        this.dataEntryFilter         = dataEntryFilter;
        this.acceptedDataEntryReader = acceptedDataEntryReader;
        this.rejectedDataEntryReader = rejectedDataEntryReader;
    }


    // Implementations for DataEntryReader.

    public void read(DataEntry dataEntry)
    throws IOException
    {
        DataEntryReader dataEntryReader = dataEntryFilter.accepts(dataEntry) ?
            acceptedDataEntryReader :
            rejectedDataEntryReader;

        if (dataEntryReader != null)
        {
            dataEntryReader.read(dataEntry);
        }
    }
}
