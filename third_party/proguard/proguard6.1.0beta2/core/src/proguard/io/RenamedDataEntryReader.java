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

import proguard.classfile.ClassConstants;

import java.io.IOException;
import java.util.Map;

/**
 * This DataEntryReader delegates to another DataEntryReader, renaming the
 * data entries based on the given map. Entries whose name does not appear
 * in the map may be passed to an alternative DataEntryReader.
 *
 * @author Eric Lafortune
 */
public class RenamedDataEntryReader implements DataEntryReader
{
    private final Map             nameMap;
    private final DataEntryReader dataEntryReader;
    private final DataEntryReader missingDataEntryReader;


    /**
     * Creates a new RenamedDataEntryReader.
     * @param nameMap         the map from old names to new names.
     * @param dataEntryReader the DataEntryReader to which renamed data
     *                        entries will be passed.
     */
    public RenamedDataEntryReader(Map             nameMap,
                                  DataEntryReader dataEntryReader)
    {
        this(nameMap, dataEntryReader, null);
    }


    /**
     * Creates a new RenamedDataEntryReader.
     * @param nameMap                the map from old names to new names.
     * @param dataEntryReader        the DataEntryReader to which renamed data
     *                               entries will be passed.
     * @param missingDataEntryReader the optional DataEntryReader to which data
     *                               entries that can't be renamed will be
     *                               passed.
     */
    public RenamedDataEntryReader(Map             nameMap,
                                  DataEntryReader dataEntryReader,
                                  DataEntryReader missingDataEntryReader)
    {
        this.nameMap                = nameMap;
        this.dataEntryReader        = dataEntryReader;
        this.missingDataEntryReader = missingDataEntryReader;
    }


    // Implementations for DataEntryReader.

    public void read(DataEntry dataEntry) throws IOException
    {
        String name = dataEntry.getName();

        // Add a directory separator if necessary.
        if (dataEntry.isDirectory() &&
            name.length() > 0)
        {
            name += ClassConstants.PACKAGE_SEPARATOR;
        }

        String newName = (String)nameMap.get(name);
        if (newName != null)
        {
            // Remove the directory separator if necessary.
            if (dataEntry.isDirectory() &&
                newName.length() > 0)
            {
                newName = newName.substring(0, newName.length() -  1);
            }

            dataEntryReader.read(new RenamedDataEntry(dataEntry, newName));
        }
        else if (missingDataEntryReader != null)
        {
            missingDataEntryReader.read(dataEntry);
        }
    }
}
