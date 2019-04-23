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
package proguard.gui;

import javax.swing.filechooser.FileFilter;
import java.io.File;


/**
 * This <code>FileFilter</code> accepts files that end in one of the given
 * extensions.
 *
 * @author Eric Lafortune
 */
final class ExtensionFileFilter extends FileFilter
{
    private final String   description;
    private final String[] extensions;


    /**
     * Creates a new ExtensionFileFilter.
     * @param description a description of the filter.
     * @param extensions  an array of acceptable extensions.
     */
    public ExtensionFileFilter(String description, String[] extensions)
    {
        this.description = description;
        this.extensions  = extensions;
    }


    // Implemntations for FileFilter

    public String getDescription()
    {
        return description;
    }


    public boolean accept(File file)
    {
        if (file.isDirectory())
        {
            return true;
        }

        String fileName = file.getName().toLowerCase();

        for (int index = 0; index < extensions.length; index++)
        {
            if (fileName.endsWith(extensions[index]))
            {
                return true;
            }
        }

        return false;
    }
}
