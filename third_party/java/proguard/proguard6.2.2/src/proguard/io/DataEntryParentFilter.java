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

/**
 * This DataEntryFilter delegates filtering to a DataEntryFilter for its parent.
 *
 * @author Eric Lafortune
 */
public class DataEntryParentFilter
implements   DataEntryFilter
{
    private final DataEntryFilter dataEntryFilter;


    /**
     * Creates a new ParentFilter.
     * @param dataEntryFilter the filter that will be applied to the data
     *                        entry's parent.
     */
    public DataEntryParentFilter(DataEntryFilter dataEntryFilter)
    {
        this.dataEntryFilter = dataEntryFilter;
    }


    // Implementations for DataEntryFilter.

    public boolean accepts(DataEntry dataEntry)
    {
        return dataEntry != null && dataEntryFilter.accepts(dataEntry.getParent());
    }
}
