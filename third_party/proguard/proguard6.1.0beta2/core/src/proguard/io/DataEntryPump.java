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
 * This interface provides a method to pump data entries. The implementation
 * determines the source and the type of the data entries. Typical examples
 * are zip entries coming from a zip file of file entries coming from a
 * directory structure. The reader can for instance collect the classes,
 * or copy the resource files that are presented.
 *
 * @author Eric Lafortune
 */
public interface DataEntryPump
{
    /**
     * Applies the given DataEntryReader to all data entries that the
     * implementation can provide.
     */
    public void pumpDataEntries(DataEntryReader dataEntryReader)
    throws IOException;
}
