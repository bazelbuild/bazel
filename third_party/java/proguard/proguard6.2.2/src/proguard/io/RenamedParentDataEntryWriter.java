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

import proguard.util.StringMatcher;

import java.io.*;

/**
 * This DataEntryWriter delegates to another DataEntryWriter, renaming
 * parent data entries based on the given matcher.
 *
 * @author Thomas Neidhart
 */
public class RenamedParentDataEntryWriter implements DataEntryWriter
{
    private final StringMatcher   matcher;
    private final String          newParentName;
    private final DataEntryWriter dataEntryWriter;


    /**
     * Creates a new RenamedParentDataEntryWriter.
     *
     * @param matcher                the string matcher to match parent entries.
     * @param newParentName          the new parent name to use.
     * @param dataEntryWriter        the DataEntryWriter to which the writing will
     *                               be delegated.
     */
    public RenamedParentDataEntryWriter(StringMatcher   matcher,
                                        String          newParentName,
                                        DataEntryWriter dataEntryWriter)
    {
        this.matcher         = matcher;
        this.newParentName   = newParentName;
        this.dataEntryWriter = dataEntryWriter;
    }


    // Implementations for DataEntryWriter.

    public boolean createDirectory(DataEntry dataEntry) throws IOException
    {
        return dataEntryWriter.createDirectory(getRedirectedEntry(dataEntry));
    }


    public boolean sameOutputStream(DataEntry dataEntry1, DataEntry dataEntry2)
        throws IOException
    {
        return dataEntryWriter.sameOutputStream(getRedirectedEntry(dataEntry1),
                                                getRedirectedEntry(dataEntry2));
    }


    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException
    {
        return dataEntryWriter.createOutputStream(getRedirectedEntry(dataEntry));
    }


    public void close() throws IOException
    {
        dataEntryWriter.close();
    }


    public void println(PrintWriter pw, String prefix)
    {
        dataEntryWriter.println(pw, prefix);
    }

    private DataEntry getRedirectedEntry(DataEntry dataEntry)
    {
        if (dataEntry == null)
        {
            return null;
        }

        final DataEntry parentEntry = dataEntry.getParent();
        if (parentEntry != null &&
            matcher.matches(parentEntry.getName()))
        {
            final DataEntry renamedParentEntry =
                new RenamedDataEntry(parentEntry, newParentName);

            return new WrappedDataEntry(dataEntry) {
                public DataEntry getParent()
                {
                    return renamedParentEntry;
                }
            };
        }

        return dataEntry;
    }

}
