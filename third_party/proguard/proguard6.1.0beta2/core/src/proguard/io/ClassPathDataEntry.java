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

import static proguard.classfile.ClassConstants.CLASS_FILE_EXTENSION;
import static proguard.classfile.util.ClassUtil.internalClassName;

/**
 * DataEntry implementation which loads an input stream from the classpath of
 * the running VM.
 *
 * @author Johan Leys
 */
public class ClassPathDataEntry implements DataEntry
{
    private final String name;

    private InputStream inputStream;


    /**
     * Creas an new ClassPathDataEntry for the given class.
     *
     * @param clazz the class for which to create a data entry.
     */
    public ClassPathDataEntry(Class clazz)
    {
        this(internalClassName(clazz.getName()) + CLASS_FILE_EXTENSION);
    }


    /**
     * Creates a new ClassPathDataEntry for the entry with the given name.
     *
     * @param name the name of the class for which to create a data entry.
     */
    public ClassPathDataEntry(String name)
    {
        this.name = name;
    }


    // Implementations for DataEntry.

    public String getName()
    {
        return name;
    }


    public String getOriginalName()
    {
        return name;
    }


    public long getSize()
    {
        return -1;
    }


    public boolean isDirectory()
    {
        return false;
    }


    public InputStream getInputStream() throws IOException
    {
        if (inputStream == null)
        {
            inputStream = getClass().getClassLoader().getResourceAsStream(name);
        }
        return inputStream;
    }


    public void closeInputStream() throws IOException
    {
        inputStream.close();
        inputStream = null;
    }


    public DataEntry getParent()
    {
        return null;
    }


    // Implementations for Object.

    public String toString()
    {
        return getName();
    }
}
